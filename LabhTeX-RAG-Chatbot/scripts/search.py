import argparse

from app.config import TOP_K, VECTOR_STORE_PATH
from app.ingestion.indexing import load_vector_store


def search_documents(
    question: str,
    top_k: int = TOP_K,
    score_threshold: float = 0.25
):
    store = load_vector_store()
    return store.search(
        question,
        top_k=top_k,
        score_threshold=score_threshold
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    if not args.query.strip():
        parser.error("Query cannot be empty")

    if not -1.0 <= args.threshold <= 1.0:
        parser.error("Threshold must be between -1 and 1")

    results = search_documents(
        args.query,
        args.top_k,
        args.threshold
    )

    if not results:
        print("No relevant documents found.")
    else:
        print("\\nSEARCH RESULTS")
        print("=" * 60)

        for rank, result in enumerate(results, start=1):
            metadata = result["metadata"]

            print(f"\\nResult #{rank}")
            print(f"Similarity: {result['score']:.4f}")
            print(f"Source: {metadata.get('source')}")
            print(f"Page: {metadata.get('page')}")
            print(f"Chunk ID: {metadata.get('chunk_id')}")
            print(f"Content: {result['text']}")
            print("-" * 60)
