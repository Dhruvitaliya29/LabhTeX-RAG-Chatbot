from pathlib import Path
import streamlit as st
import pymupdf

from app.config import (
    APP_NAME, DOCUMENT_PATH, TOP_K, SCORE_THRESHOLD
)
from app.ingestion.indexing import build_index, load_vector_store
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generation.llm import GroqGenerator


st.set_page_config(
    page_title=APP_NAME,
    page_icon="📚",
    layout="wide"
)

st.title("LabhTeX RAG Chatbot")
st.caption("Ask questions using information from your uploaded documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

if "generator" not in st.session_state:
    st.session_state.generator = None


with st.sidebar:
    st.header("Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("Build / Update Index"):
        if not uploaded_files:
            st.warning("Upload at least one document.")
        else:
            valid_files = []
            rejected_files = []

            for file in uploaded_files:
                filename = Path(file.name).name
                file_data = file.getvalue()

                try:
                    if filename.lower().endswith(".pdf"):
                        with pymupdf.open(
                            stream=file_data,
                            filetype="pdf"
                        ) as pdf:
                            text_chars = sum(
                                len(page.get_text().strip())
                                for page in pdf
                            )

                        if text_chars == 0:
                            rejected_files.append(
                                (filename, "No selectable text found. Scanned PDFs are not supported.")
                            )
                        else:
                            valid_files.append((filename, file_data))

                    elif filename.lower().endswith(".txt"):
                        text = file_data.decode("utf-8").strip()

                        if not text:
                            rejected_files.append(
                                (filename, "The text file is empty.")
                            )
                        else:
                            valid_files.append((filename, file_data))

                except Exception as exc:
                    rejected_files.append((filename, str(exc)))

            for filename, reason in rejected_files:
                st.warning("{}: {}".format(filename, reason))

            if valid_files:
                try:
                    DOCUMENT_PATH.mkdir(parents=True, exist_ok=True)

                    for filename, file_data in valid_files:
                        target = DOCUMENT_PATH / filename
                        target.write_bytes(file_data)

                    with st.spinner("Building index..."):
                        result = build_index()

                    st.session_state.index_ready = True
                    st.success(
                        "Indexed {} documents and {} chunks.".format(
                            result["documents"],
                            result["chunks"]
                        )
                    )

                except Exception as exc:
                    st.error("Index building failed: {}".format(exc))

            elif rejected_files:
                st.error(
                    "No valid documents were uploaded. Existing index was not rebuilt."
                )

    if st.button("Load Existing Index"):
        try:
            store = load_vector_store()
            st.session_state.index_ready = store.index.ntotal > 0
            st.success("Loaded {} chunks.".format(store.index.ntotal))
        except Exception as exc:
            st.session_state.index_ready = False
            st.error("Loading failed: {}".format(exc))

    retrieval_mode = st.selectbox(
        "Retrieval method",
        ["Semantic Search (FAISS)", "Hybrid Search (FAISS + BM25)"]
    )

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Embeddings: all-MiniLM-L6-v2")
    st.caption("Vector database: FAISS")
    st.caption("Language model: Groq")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources and retrieved context"):
                for source in message["sources"]:
                    st.markdown(
                        "**[Source {}]** {}".format(
                            source["citation"], source["source"]
                        )
                    )
                    if source.get("page"):
                        st.caption("Page: {}".format(source["page"]))
                    st.caption("Chunk: {}".format(source["chunk_id"]))
                    st.write(source.get("text", ""))


question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not st.session_state.index_ready:
            st.warning("Please load or build the knowledge base first.")
        else:
            try:
                with st.spinner("Searching documents and generating answer..."):
                    store = load_vector_store()

                    if retrieval_mode == "Hybrid Search (FAISS + BM25)":
                        retriever = HybridRetriever(store)
                        results = retriever.search(
                            question,
                            top_k=TOP_K,
                            score_threshold=SCORE_THRESHOLD
                        )
                    else:
                        results = store.search(
                            question,
                            top_k=TOP_K,
                            score_threshold=SCORE_THRESHOLD
                        )

                    if st.session_state.generator is None:
                        st.session_state.generator = GroqGenerator()

                    response = st.session_state.generator.generate_answer(
                        question, results
                    )

                st.markdown(response["answer"])

                sources = []

                for source in response["sources"]:
                    match = next(
                        (
                            item for item in results
                            if item["metadata"].get("chunk_id") == source["chunk_id"]
                        ),
                        None
                    )

                    item = dict(source)
                    item["text"] = match["text"] if match else ""
                    sources.append(item)

                if sources:
                    with st.expander("Sources and retrieved context"):
                        for source in sources:
                            st.markdown(
                                "**[Source {}]** {}".format(
                                    source["citation"], source["source"]
                                )
                            )
                            if source.get("page"):
                                st.caption("Page: {}".format(source["page"]))
                            st.caption("Chunk: {}".format(source["chunk_id"]))
                            st.write(source["text"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": sources
                })

            except Exception as exc:
                st.error("An error occurred: {}".format(exc))
