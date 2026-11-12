from app.ingestion.indexing import build_index


if __name__ == "__main__":
    print("Starting document indexing...")

    result = build_index()

    print("\\nIndexing completed successfully!")
    for key, value in result.items():
        print(f"{key}: {value}")
