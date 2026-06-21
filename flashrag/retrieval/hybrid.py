"""
Hybrid dense + sparse search with reciprocal rank fusion.

Combines FAISS vector search with BM25 keyword search to get the best
of both worlds: semantic understanding and exact keyword matching.
"""

from __future__ import annotations

import logging
from typing import Any

from flashrag.embeddings.base import BaseEmbedding
from flashrag.registry import RETRIEVERS
from flashrag.retrieval.bm25 import BM25Retriever
from flashrag.retrieval.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[SearchResult]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score = sum over lists of  weight / (k + rank_in_list)

    Parameters
    ----------
    result_lists : list of list of SearchResult
        Multiple ranked result sets to fuse.
    k : int
        RRF constant (higher = smoother fusion).
    weights : list of float, optional
        Per-list weight multipliers (defaults to uniform).
    """
    if weights is None:
        weights = [1.0] * len(result_lists)

    doc_scores: dict[str, float] = {}
    doc_map: dict[str, SearchResult] = {}

    for results, weight in zip(result_lists, weights):
        for rank, result in enumerate(results):
            key = result.text[:200]
            rrf_score = weight / (k + rank + 1)
            doc_scores[key] = doc_scores.get(key, 0.0) + rrf_score
            if key not in doc_map or result.score > doc_map[key].score:
                doc_map[key] = result

    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

    fused: list[SearchResult] = []
    for key in sorted_keys:
        r = doc_map[key]
        fused.append(
            SearchResult(
                text=r.text,
                score=doc_scores[key],
                metadata=r.metadata,
                vector_id=r.vector_id,
            )
        )
    return fused


@RETRIEVERS.register("hybrid")
class HybridSearch:
    """
    Combined dense (FAISS) + sparse (BM25) retrieval with RRF fusion.

    Parameters
    ----------
    embedding_model : BaseEmbedding or str
        Embedding model for dense retrieval. If a string, loads a
        SentenceTransformerEmbedding with that model name.
    alpha : float
        Weight for dense results vs sparse. 1.0 = all dense, 0.0 = all sparse.
    rrf_k : int
        RRF fusion constant.
    """

    def __init__(
        self,
        embedding_model: BaseEmbedding | str = "all-MiniLM-L6-v2",
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        if isinstance(embedding_model, str):
            from flashrag.embeddings.sentence_transformer import SentenceTransformerEmbedding

            self._embedder = SentenceTransformerEmbedding(embedding_model)
        else:
            self._embedder = embedding_model

        self.alpha = alpha
        self.rrf_k = rrf_k

        self._vector_store = VectorStore(dimension=self._embedder.dimension, metric="cosine")
        self._bm25 = BM25Retriever()

        self._documents: list[str] = []
        self._metadata: list[dict[str, Any]] = []

    @property
    def size(self) -> int:
        return len(self._documents)

    def index(
        self,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Index documents for both dense and sparse search."""
        self._documents = list(documents)
        self._metadata = list(metadata) if metadata else [{}] * len(documents)

        logger.info(f"Encoding {len(documents)} documents for dense index...")
        vectors = self._embedder.encode(documents, show_progress=True)
        self._vector_store.add(vectors, documents, self._metadata)

        logger.info("Building BM25 sparse index...")
        self._bm25.index(documents, self._metadata)

        logger.info(f"Hybrid index built: {self.size} documents")

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Search using both dense and sparse retrieval, fused with RRF.
        """
        fetch_k = max(top_k * 3, 20)
        dense_k = dense_top_k or fetch_k
        sparse_k = sparse_top_k or fetch_k

        query_vec = self._embedder.encode([query])[0]
        dense_results = self._vector_store.search(query_vec, top_k=dense_k)
        sparse_results = self._bm25.search(query, top_k=sparse_k)

        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self.rrf_k,
            weights=[self.alpha, 1.0 - self.alpha],
        )

        return fused[:top_k]

    def add_documents(
        self,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Incrementally add documents to both indices."""
        meta = list(metadata) if metadata else [{}] * len(documents)
        vectors = self._embedder.encode(documents)
        self._vector_store.add(vectors, documents, meta)
        self._bm25.add_documents(documents, meta)
        self._documents.extend(documents)
        self._metadata.extend(meta)
