from pathlib import Path
import pytest

from app.ingestion.document_loader import load_document, clean_text
from app.ingestion.text_splitter import split_text, create_chunks


def test_clean_text():
    assert clean_text("Hello   World\nTest") == "Hello World Test"


def test_split_text():
    text = " ".join(["word"] * 300)
    chunks = split_text(text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_empty_text():
    assert split_text("") == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        split_text("Hello world", chunk_size=0)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("Hello world", chunk_size=100, chunk_overlap=100)


def test_txt_loading(tmp_path):
    file = tmp_path / "sample.txt"
    file.write_text("RAG combines retrieval and generation.", encoding="utf-8")

    documents = load_document(str(file))

    assert len(documents) == 1
    assert documents[0]["source"] == "sample.txt"
    assert documents[0]["page"] is None


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_document("missing.txt")


def test_chunk_metadata():
    documents = [{
        "text": "This is a sample document about RAG.",
        "source": "sample.txt",
        "page": None
    }]

    chunks = create_chunks(documents, 20, 5)

    assert len(chunks) > 0
    assert chunks[0]["source"] == "sample.txt"
    assert "chunk_id" in chunks[0]
