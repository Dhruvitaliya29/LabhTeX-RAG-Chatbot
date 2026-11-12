import re
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class HybridRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.documents = vector_store.metadata

        if not self.documents:
            raise ValueError("No indexed documents available.")

        tokenized_documents = [
            tokenize(doc.get("text", ""))
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.25,
        candidate_k: int = 15,
        rrf_constant: int = 60,
        bm25_threshold: float = 1.0,
    ) -> list[dict]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        semantic_results = self.vector_store.search(
            query,
            top_k=min(candidate_k, len(self.documents))
        )

        query_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:candidate_k]

        bm25_results = [
            {
                "index": idx,
                "score": float(bm25_scores[idx])
            }
            for idx in bm25_indices
            if bm25_scores[idx] > 0
        ]

        semantic_rank = {
            result["metadata"]["chunk_id"]: rank
            for rank, result in enumerate(semantic_results, start=1)
        }

        bm25_rank = {
            self.documents[item["index"]]["chunk_id"]: rank
            for rank, item in enumerate(bm25_results, start=1)
        }

        semantic_scores = {
            result["metadata"]["chunk_id"]: result["score"]
            for result in semantic_results
        }

        bm25_raw_scores = {
            self.documents[item["index"]]["chunk_id"]: item["score"]
            for item in bm25_results
        }

        candidates = set(semantic_rank) | set(bm25_rank)

        # Reject queries with no meaningful semantic or lexical evidence.
        max_semantic = max(semantic_scores.values(), default=-1.0)
        max_bm25 = max(bm25_raw_scores.values(), default=0.0)

        if max_semantic < score_threshold and max_bm25 < bm25_threshold:
            return []

        fused_results = []

        for chunk_id in candidates:
            semantic_position = semantic_rank.get(chunk_id)
            lexical_position = bm25_rank.get(chunk_id)

            rrf_score = 0.0

            if semantic_position is not None:
                rrf_score += 1 / (rrf_constant + semantic_position)

            if lexical_position is not None:
                rrf_score += 1 / (rrf_constant + lexical_position)

            document = next(
                doc for doc in self.documents
                if doc["chunk_id"] == chunk_id
            )

            fused_results.append({
                "text": document["text"],
                "score": semantic_scores.get(chunk_id, 0.0),
                "hybrid_score": rrf_score,
                "bm25_score": bm25_raw_scores.get(chunk_id, 0.0),
                "metadata": {
                    key: value
                    for key, value in document.items()
                    if key != "text"
                }
            })

        fused_results.sort(
            key=lambda item: item["hybrid_score"],
            reverse=True
        )

        return fused_results[:top_k]
