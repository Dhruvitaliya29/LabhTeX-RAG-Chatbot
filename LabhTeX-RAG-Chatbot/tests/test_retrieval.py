import numpy as np
import pytest

from app.retrieval.vector_store import FaissVectorStore


class FakeEmbeddingModel:
    vectors = {
        "machine learning": [1, 0, 0],
        "deep learning": [0, 1, 0],
        "database systems": [0, 0, 1],
        "ai models": [0.9, 0.1, 0],
    }

    def encode_documents(self, texts):
        return np.array(
            [self._vector(text) for text in texts],
            dtype=np.float32
        )

    def encode_query(self, query):
        return np.array(
            [self._vector(query)],
            dtype=np.float32
        )

    def _vector(self, text):
        return np.array(
            self.vectors.get(text, [1, 0, 0]),
            dtype=np.float32
        )


@pytest.fixture
def store():
    model = FakeEmbeddingModel()
    vector_store = FaissVectorStore(model)

    documents = [
        {
            "text": "machine learning",
            "source": "ml.txt",
            "page": 1,
            "chunk_id": "ml_1"
        },
        {
            "text": "deep learning",
            "source": "dl.txt",
            "page": 2,
            "chunk_id": "dl_1"
        },
        {
            "text": "database systems",
            "source": "db.txt",
            "page": 1,
            "chunk_id": "db_1"
        }
    ]

    vector_store.add_documents(documents)
    return vector_store


def test_add_documents(store):
    assert store.index.ntotal == 3
    assert len(store.metadata) == 3


def test_semantic_search(store):
    results = store.search("machine learning", top_k=2)

    assert len(results) == 2
    assert results[0]["metadata"]["source"] == "ml.txt"
    assert results[0]["score"] == pytest.approx(1.0)


def test_top_k_limit(store):
    results = store.search("machine learning", top_k=1)
    assert len(results) == 1


def test_search_empty_store():
    store = FaissVectorStore(FakeEmbeddingModel())
    assert store.search("machine learning") == []


def test_save_and_load(store, tmp_path):
    store.save(str(tmp_path))

    new_store = FaissVectorStore(FakeEmbeddingModel())
    new_store.load(str(tmp_path))

    assert new_store.index.ntotal == 3
    assert len(new_store.metadata) == 3

    results = new_store.search("deep learning", top_k=1)

    assert results[0]["metadata"]["source"] == "dl.txt"


def test_invalid_top_k(store):
    with pytest.raises(ValueError):
        store.search("machine learning", top_k=0)


def test_empty_documents():
    store = FaissVectorStore(FakeEmbeddingModel())

    with pytest.raises(ValueError):
        store.add_documents([])


def test_dimension_mismatch():
    store = FaissVectorStore(FakeEmbeddingModel())

    documents = [
        {"text": "machine learning"},
        {"text": "deep learning"}
    ]

    store.add_documents(documents)

    class WrongModel:
        def encode_query(self, query):
            return np.array([[1, 0]], dtype=np.float32)

        def encode_documents(self, texts):
            return np.array([[1, 0]], dtype=np.float32)

    store.embedding_model = WrongModel()

    with pytest.raises(ValueError):
        store.search("machine learning")
