import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

APP_NAME = "LabhTeX RAG Chatbot"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DOCUMENT_PATH = PROJECT_ROOT / "data" / "documents"
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

TOP_K = 5
SCORE_THRESHOLD = 0.25
