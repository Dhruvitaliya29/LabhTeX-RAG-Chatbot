import json
from pathlib import Path

import faiss
import numpy as np


class FaissVectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.metadata = []

    def add_documents(self, documents: list[dict]):
        if not documents:
            raise ValueError("No documents provided")

        valid_documents = [
            doc for doc in documents
            if doc.get("text", "").strip()
        ]

        if not valid_documents:
            raise ValueError("No valid document text found")

        texts = [doc["text"] for doc in valid_documents]

        embeddings = self.embedding_model.encode_documents(texts)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
            raise ValueError("Invalid embedding dimensions")

        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])

        elif self.index.d != embeddings.shape[1]:
            raise ValueError("Embedding dimension mismatch")

        self.index.add(embeddings)
        self.metadata.extend(valid_documents)

        return len(valid_documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None
    ) -> list[dict]:

        if self.index is None or self.index.ntotal == 0:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = self.embedding_model.encode_query(query)
        query_embedding = np.ascontiguousarray(
            query_embedding, dtype=np.float32
        )

        if query_embedding.shape != (1, self.index.d):
            raise ValueError("Query embedding dimension mismatch")

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or (score_threshold is not None and float(score) < score_threshold):
                continue

            results.append({
                "text": self.metadata[idx]["text"],
                "score": float(score),
                "metadata": {
                    key: value
                    for key, value in self.metadata[idx].items()
                    if key != "text"
                }
            })

        return results

    def save(self, directory: str = "vector_store"):
        if self.index is None:
            raise ValueError("No index to save")

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(path / "index.faiss")
        )

        with open(path / "metadata.json", "w", encoding="utf-8") as file:
            json.dump(self.metadata, file, ensure_ascii=False, indent=2)

    def load(self, directory: str = "vector_store"):
        path = Path(directory)

        index_path = path / "index.faiss"
        metadata_path = path / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("FAISS index or metadata not found")

        index = faiss.read_index(str(index_path))

        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        if index.ntotal != len(metadata):
            raise ValueError("Index and metadata count mismatch")

        self.index = index
        self.metadata = metadata
