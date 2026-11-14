# RAG Chatbot Architecture

## System Workflow

```mermaid
flowchart TD
    A["PDF / TXT Documents"] --> B["Document Loader"]
    B --> C["Text Extraction"]
    C --> D["Sentence-Aware Chunking"]
    D --> E["Embedding Model: all-MiniLM-L6-v2"]
    E --> F["FAISS Vector Index"]
    F --> G["Stored Chunks and Metadata"]
    H["User Question"] --> I["Streamlit Interface"]
    I --> J["Retrieval Selection"]
    J --> K["FAISS Semantic Search"]
    J --> L["BM25 Keyword Search"]
    K --> M["Retrieved Context"]
    L --> N["Reciprocal Rank Fusion"]
    N --> M
    M --> O["Groq LLM"]
    H --> O
    O --> P["Grounded Answer"]
    O --> Q["Source Citations"]
    O --> R["Unsupported Question Refusal"]
    P --> I
    Q --> I
    R --> I
```

## Main Components

- **Document ingestion:** Extracts text from PDF and TXT files.
- **Chunking:** Splits extracted text into manageable, overlapping chunks.
- **Embeddings:** Converts document chunks and user questions into 384-dimensional vectors.
- **Vector database:** FAISS IndexFlatIP stores and searches normalized embeddings.
- **Hybrid retrieval:** Combines BM25 keyword retrieval and FAISS semantic retrieval using Reciprocal Rank Fusion.
- **LLM:** Groq generates responses using retrieved document context.
- **Interface:** Streamlit provides document upload, index management, retrieval selection, chat history and source display.

## Answer Grounding

The LLM receives the user question and retrieved context. It is instructed to answer using the supplied context, cite sources and refuse questions when sufficient supporting information is unavailable.
