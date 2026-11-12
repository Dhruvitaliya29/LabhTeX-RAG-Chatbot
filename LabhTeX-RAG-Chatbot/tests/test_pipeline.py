from app.ingestion.pipeline import ingest_documents


def test_ingestion_pipeline(tmp_path):
    file = tmp_path / "rag.txt"
    file.write_text(
        "Retrieval augmented generation uses external documents "
        "to provide context to language models. " * 30,
        encoding="utf-8"
    )

    chunks = ingest_documents(str(tmp_path))

    assert len(chunks) > 1
    assert all(chunk["source"] == "rag.txt" for chunk in chunks)
    assert all("chunk_id" in chunk for chunk in chunks)
    assert all(chunk["text"] for chunk in chunks)
