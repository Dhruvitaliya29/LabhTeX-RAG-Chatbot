from pathlib import Path
import pymupdf as fitz
import re

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_document(file_path: str) -> list[dict]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    documents = []

    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = clean_text(page.get_text())

                if text:
                    documents.append({
                        "text": text,
                        "source": path.name,
                        "page": page_number
                    })

        if not documents:
            raise ValueError(
                f"No selectable text found in PDF: {path.name}. "
                "Scanned/image-only PDFs are not supported."
            )

    elif path.suffix.lower() == ".txt":
        text = clean_text(path.read_text(encoding="utf-8"))

        if text:
            documents.append({
                "text": text,
                "source": path.name,
                "page": None
            })

    if not documents:
        raise ValueError(f"No readable text found in document: {path.name}")

    return documents


def load_directory(directory: str) -> list[dict]:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    all_documents = []

    for file_path in sorted(path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            all_documents.extend(load_document(str(file_path)))

    return all_documents
