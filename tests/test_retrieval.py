"""
Tests for retrieval components: VectorStore, BM25, HybridSearch.
"""

import numpy as np
import pytest

from flashrag.retrieval.bm25 import BM25Retriever
from flashrag.retrieval.vector_store import VectorStore


class TestVectorStore:
    def test_create_empty_store(self):
        store = VectorStore(dimension=4, metric="cosine")
        assert store.size == 0
        assert store.dimension == 4

    def test_add_and_search(self):
        store = VectorStore(dimension=4, metric="cosine")
        vectors = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]],
            dtype=np.float32,
        )
        docs = ["doc_a", "doc_b", "doc_c"]
        store.add(vectors, docs)
        assert store.size == 3

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=2)
        assert len(results) == 2
        assert results[0].text == "doc_a"
        assert results[0].score > results[1].score

    def test_search_empty_store(self):
        store = VectorStore(dimension=4, metric="cosine")
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=5)
        assert results == []

    def test_metadata_preserved(self):
        store = VectorStore(dimension=2)
        vectors = np.array([[1.0, 0.0]], dtype=np.float32)
        store.add(vectors, ["test_doc"], [{"source": "test.txt", "page": 1}])
        results = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
        assert results[0].metadata["source"] == "test.txt"
        assert results[0].metadata["page"] == 1

    def test_l2_metric(self):
        store = VectorStore(dimension=3, metric="l2")
        vectors = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        store.add(vectors, ["x", "y", "z"])
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=1)
        assert results[0].text == "x"

    def test_dimension_mismatch_raises(self):
        store = VectorStore(dimension=4)
        bad_vectors = np.array([[1.0, 0.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="dimension"):
            store.add(bad_vectors, ["doc"])

    def test_count_mismatch_raises(self):
        store = VectorStore(dimension=2)
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="count"):
            store.add(vectors, ["only_one"])

    def test_save_and_load(self, tmp_path):
        store = VectorStore(dimension=3, metric="cosine")
        vectors = np.random.randn(5, 3).astype(np.float32)
        docs = [f"doc_{i}" for i in range(5)]
        meta = [{"idx": i} for i in range(5)]
        store.add(vectors, docs, meta)

        store.save(tmp_path / "test_index")
        loaded = VectorStore.load(tmp_path / "test_index")

        assert loaded.size == 5
        assert loaded.dimension == 3

    def test_batch_search(self):
        store = VectorStore(dimension=3)
        vectors = np.eye(3, dtype=np.float32)
        store.add(vectors, ["x", "y", "z"])

        queries = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        batch_results = store.batch_search(queries, top_k=1)
        assert len(batch_results) == 2
        assert batch_results[0][0].text == "x"
        assert batch_results[1][0].text == "y"


class TestBM25Retriever:
    def test_empty_retriever(self):
        bm25 = BM25Retriever()
        assert bm25.size == 0
        results = bm25.search("test query")
        assert results == []

    def test_index_and_search(self):
        bm25 = BM25Retriever()
        docs = [
            "The cat sat on the mat",
            "Dogs are loyal companions",
            "Machine learning transforms data into knowledge",
        ]
        bm25.index(docs)
        assert bm25.size == 3

        results = bm25.search("cat mat", top_k=2)
        assert len(results) > 0
        assert results[0].text == docs[0]

    def test_search_relevance(self):
        bm25 = BM25Retriever()
        docs = [
            "Python programming language",
            "JavaScript web development",
            "Python data science and machine learning",
        ]
        bm25.index(docs)

        results = bm25.search("Python programming", top_k=3)
        assert results[0].text == docs[0]

    def test_add_documents(self):
        bm25 = BM25Retriever()
        bm25.index(["first document"])
        assert bm25.size == 1

        bm25.add_documents(["second document", "third document"])
        assert bm25.size == 3

    def test_metadata_preserved(self):
        bm25 = BM25Retriever()
        bm25.index(
            ["test document"],
            metadata=[{"source": "test.txt"}],
        )
        results = bm25.search("test", top_k=1)
        assert results[0].metadata["source"] == "test.txt"
