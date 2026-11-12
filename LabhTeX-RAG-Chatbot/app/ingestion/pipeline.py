from app.config import DOCUMENT_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from app.ingestion.document_loader import load_directory
from app.ingestion.text_splitter import create_chunks


def ingest_documents(directory=DOCUMENT_PATH):
    documents = load_directory(directory)

    chunks = create_chunks(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    if not chunks:
        raise ValueError("No document chunks generated")

    return chunks
