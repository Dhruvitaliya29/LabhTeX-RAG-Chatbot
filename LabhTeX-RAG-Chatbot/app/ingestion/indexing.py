from app.config import (
    DOCUMENT_PATH,
    VECTOR_STORE_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from app.ingestion.document_loader import load_directory
from app.ingestion.text_splitter import create_chunks
from app.embeddings.embedding_model import EmbeddingModel
from app.retrieval.vector_store import FaissVectorStore


def build_index():
    documents = load_directory(str(DOCUMENT_PATH))

    if not documents:
        raise ValueError("No documents found for indexing.")

    chunks = create_chunks(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    if not chunks:
        raise ValueError("No chunks generated.")

    embedding_model = EmbeddingModel(EMBEDDING_MODEL)
    store = FaissVectorStore(embedding_model)

    count = store.add_documents(chunks)
    store.save(str(VECTOR_STORE_PATH))

    return {
        "documents": len(documents),
        "chunks": count,
        "dimension": store.index.d,
        "path": str(VECTOR_STORE_PATH),
    }


def load_vector_store():
    embedding_model = EmbeddingModel(EMBEDDING_MODEL)
    store = FaissVectorStore(embedding_model)
    store.load(str(VECTOR_STORE_PATH))
    return store
