# LabhTeX RAG Chatbot

A document-based AI chatbot built using Retrieval-Augmented Generation (RAG). It allows users to upload PDF and TXT documents, retrieve relevant information, and receive context-grounded answers with source references.

## Features

- Upload PDF and TXT documents.
- Extract and clean document text.
- Split documents into manageable chunks.
- Generate embeddings using Sentence Transformers.
- Store and search embeddings using FAISS.
- Generate answers using the Groq API.
- Display source filenames, page numbers, and retrieved context.
- Handle questions when sufficient information is unavailable.
- Evaluate the chatbot using direct, combined, and unsupported questions.
- Experimental hybrid retrieval using BM25, FAISS, and Reciprocal Rank Fusion.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| LLM Provider | Groq |
| Language Model | openai/gpt-oss-20b |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Keyword Retrieval | BM25 |
| Document Processing | PyMuPDF |
| Evaluation | Custom Python scripts |

## Architecture

The application follows this workflow:

1. User uploads PDF or TXT documents.
2. The document loader extracts and cleans the text.
3. The text splitter divides content into smaller chunks.
4. Sentence Transformers converts chunks into embeddings.
5. FAISS stores the embeddings and associated metadata.
6. The user's question is converted into an embedding.
7. FAISS retrieves relevant document chunks.
8. The retrieved context and question are passed to the Groq LLM.
9. The chatbot generates a context-grounded answer and displays its sources.

The experimental hybrid retrieval pipeline additionally combines FAISS semantic rankings and BM25 keyword rankings using Reciprocal Rank Fusion.

## Project Structure

```text
LabhTeX-RAG-Chatbot/
│
├── app/
│   ├── config.py
│   ├── ingestion/
│   │   ├── document_loader.py
│   │   ├── text_splitter.py
│   │   └── indexing.py
│   ├── embeddings/
│   │   └── embedding_model.py
│   ├── retrieval/
│   │   ├── vector_store.py
│   │   └── hybrid_retriever.py
│   └── generation/
│       └── llm.py
│
├── data/
│   └── documents/
│
├── vector_store/
│
├── scripts/
│   ├── build_index.py
│   ├── search.py
│   ├── evaluate_rag.py
│   └── evaluate_hybrid.py
│
├── docs/
│
├── tests/
│
├── streamlit_app.py
├── requirements.txt
├── .env.example
└── README.md
```

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/Dhruvitaliya29/LabhTeX-RAG-Chatbot.git
cd LabhTeX-RAG-Chatbot
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Configuration

Create a `.env` file in the project root and configure your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Never commit your API key or `.env` file to GitHub.

## Running the Application

Start the Streamlit application:

```bash
streamlit run streamlit_app.py
```

Upload PDF or TXT documents through the sidebar and select **Add documents and rebuild index**.

Once indexing is complete, ask questions through the chat interface. The chatbot displays its answers and supporting document sources.

## Building the Index Through CLI

```bash
python scripts/build_index.py
```

## Evaluation

Run the baseline semantic retrieval evaluation:

```bash
python scripts/evaluate_rag.py
```

Run the experimental hybrid retrieval evaluation:

```bash
python scripts/evaluate_hybrid.py
```

The evaluation uses ten questions covering direct questions, combined questions, and unsupported questions.

Results are stored in the `docs/` directory.

## Improvements

The project explores the following improvements:

1. Hybrid Retrieval: Combining semantic retrieval using FAISS with keyword-based retrieval using BM25.
2. Grounded Answer Generation: Strengthening instructions to reduce unsupported answers and improve faithfulness.

The hybrid retrieval implementation is currently evaluated separately from the live Streamlit interface.

## Current Limitations

- Answer quality depends on document quality and retrieval relevance.
- Similarity-based retrieval can return irrelevant chunks.
- LLM-generated answers may still contain unsupported claims.
- Hybrid retrieval has not yet demonstrated higher answer accuracy in the current ten-question evaluation.
- Evaluation is based on a small manually reviewed dataset.
- Response latency depends on embedding generation, API response time, and network conditions.

## Future Scope

- Integrate hybrid retrieval into the live chatbot.
- Implement reranking for improved retrieval relevance.
- Improve citation validation and answer faithfulness.
- Add document metadata filtering.
- Introduce caching and performance optimization.
- Expand the evaluation dataset.

## Author

Developed as an AI/ML engineering project.
