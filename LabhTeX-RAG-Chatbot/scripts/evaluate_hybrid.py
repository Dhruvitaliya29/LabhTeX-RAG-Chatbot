import json
import time
from pathlib import Path

from app.config import TOP_K, SCORE_THRESHOLD
from app.ingestion.indexing import load_vector_store
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generation.llm import GroqGenerator

QUESTIONS = [
    ("Q1", "What is Retrieval Augmented Generation?", "Direct"),
    ("Q2", "What are the main applications of RAG?", "Direct"),
    ("Q3", "What happens during document ingestion?", "Direct"),
    ("Q4", "What are embeddings and how do they help retrieval?", "Direct"),
    ("Q5", "How does cosine similarity work with FAISS?", "Direct"),
    ("Q6", "Why does normalized inner product represent cosine similarity in FAISS?", "Combined"),
    ("Q7", "Explain the complete process from a user question to an LLM-generated answer.", "Combined"),
    ("Q8", "Which metrics evaluate a RAG system, and what does faithfulness measure?", "Combined"),
    ("Q9", "What is the population of Mars?", "Unsupported"),
    ("Q10", "Who developed the FAISS library and in which year?", "Unsupported"),
]


def main():
    store = load_vector_store()
    retriever = HybridRetriever(store)
    generator = GroqGenerator()
    records = []

    print("Starting hybrid RAG evaluation")
    print("Indexed chunks:", store.index.ntotal)

    for qid, question, category in QUESTIONS:
        print("\\n" + "=" * 65)
        print(qid, "|", category)
        print("Question:", question)

        start = time.perf_counter()

        results = retriever.search(
            question,
            top_k=TOP_K,
            score_threshold=SCORE_THRESHOLD
        )

        response = generator.generate_answer(question, results)
        elapsed = round(time.perf_counter() - start, 2)

        record = {
            "id": qid,
            "question": question,
            "category": category,
            "retrieved_chunks": len(results),
            "retrieved_sources": [
                item["metadata"].get("source") for item in results
            ],
            "semantic_scores": [
                round(item["score"], 4) for item in results
            ],
            "hybrid_scores": [
                round(item["hybrid_score"], 6) for item in results
            ],
            "answer": response["answer"],
            "sources": response["sources"],
            "latency_seconds": elapsed,
        }

        records.append(record)

        print("Retrieved chunks:", len(results))
        print("Answer:", response["answer"])
        print("Sources:", record["retrieved_sources"])
        print("Hybrid scores:", record["hybrid_scores"])
        print("Latency:", elapsed, "seconds")

    output = Path("docs")
    output.mkdir(exist_ok=True)

    destination = output / "hybrid_evaluation_results.json"
    destination.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\\nHybrid evaluation completed.")
    print("Results saved to:", destination)


if __name__ == "__main__":
    main()
