
A Retrieval-Augmented Generation (RAG) chatbot that answers questions from uploaded PDF and TXT documents. It combines semantic search, optional hybrid retrieval, and Groq-powered answer generation to provide context-grounded responses with source citations.

## Features

- PDF and TXT document ingestion
- Page-aware PDF text extraction
- Sentence-aware text chunking
- Text embeddings using Sentence Transformers
- FAISS vector indexing and similarity search
- Optional hybrid retrieval using BM25, FAISS, and Reciprocal Rank Fusion (RRF)
- Groq LLM integration for answer generation
- Source citations in generated answers
- Refusal when retrieved context does not sufficiently support an answer
- Streamlit chat interface
- Automated tests and evaluation scripts

## Architecture

The application follows this workflow:

1. Documents are uploaded or placed in the document directory.
2. The ingestion pipeline extracts and cleans document text.
3. Text is divided into overlapping, sentence-aware chunks.
4. Sentence Transformers generates embeddings.
5. FAISS stores and searches the document vectors.
6. The selected retrieval method finds relevant context.
7. Groq generates an answer using the retrieved context.
8. The interface displays the answer and its source citations.

See [Architecture Diagram](docs/architecture.md) for the workflow diagram.

## Technology Stack

- Python
- Streamlit
- Groq API
- Sentence Transformers
- FAISS CPU
- BM25 (`rank-bm25`)
- PyMuPDF
- NumPy
- Pytest

## Project Structure

```text
LabhTeX-RAG-Chatbot/
├── app/
│   ├── config.py
│   ├── main.py
│   ├── embeddings/
│   ├── generation/
│   ├── ingestion/
│   └── retrieval/
├── data/
│   └── documents/
├── docs/
│   ├── architecture.md
│   ├── baseline_evaluation.md
│   ├── evaluation_raw_results.json
│   ├── hybrid_evaluation_results.json
│   └── improvement_report.md
├── scripts/
│   ├── build_index.py
│   ├── search.py
│   ├── evaluate_rag.py
│   └── evaluate_hybrid.py
├── tests/
├── streamlit_app.py
├── requirements.txt
├── .env.example
└── README.md
