"""
RAG system validator.

Evaluates retrieval quality and generation accuracy on validation datasets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from flashrag.analytics.metrics import compute_mrr, compute_ndcg, compute_recall_at_k
from flashrag.embeddings.base import BaseEmbedding
from flashrag.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGValidator:
    """
    Validate a RAG system on a held-out evaluation set.

    Computes retrieval metrics (Recall@K, MRR, NDCG) and optionally
    generation quality metrics.

    Parameters
    ----------
    embedding_model : BaseEmbedding
        Embedding model for encoding queries.
    vector_store : VectorStore
        Pre-built vector store to evaluate against.
    ks : list of int
        K values for Recall@K and NDCG@K metrics.
    """

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        vector_store: VectorStore,
        ks: list[int] | None = None,
    ) -> None:
        self._embedder = embedding_model
        self._store = vector_store
        self.ks = ks or [1, 3, 5, 10]

    def evaluate_retrieval(
        self,
        queries: list[str],
        relevant_docs: list[list[str]],
        top_k: int = 10,
    ) -> dict[str, float]:
        """
        Evaluate retrieval quality.

        Parameters
        ----------
        queries : list of str
            Evaluation queries.
        relevant_docs : list of list of str
            Ground-truth relevant document texts for each query.
        top_k : int
            Maximum number of results to retrieve per query.
        """
        all_retrieved: list[list[str]] = []

        for query in queries:
            query_vec = self._embedder.encode([query])[0]
            results = self._store.search(query_vec, top_k=top_k)
            retrieved_texts = [r.text for r in results]
            all_retrieved.append(retrieved_texts)

        metrics: dict[str, float] = {}
        for k in self.ks:
            if k <= top_k:
                recall = compute_recall_at_k(all_retrieved, relevant_docs, k)
                metrics[f"recall@{k}"] = recall

        metrics["mrr"] = compute_mrr(all_retrieved, relevant_docs)

        for k in self.ks:
            if k <= top_k:
                ndcg = compute_ndcg(all_retrieved, relevant_docs, k)
                metrics[f"ndcg@{k}"] = ndcg

        logger.info(f"Retrieval evaluation: {metrics}")
        return metrics

    def evaluate_from_file(
        self,
        eval_path: str | Path,
        top_k: int = 10,
    ) -> dict[str, float]:
        """
        Evaluate from a JSONL file with ``{"query": ..., "relevant": [...]}`` entries.
        """
        queries: list[str] = []
        relevant: list[list[str]] = []

        with open(eval_path) as f:
            for line in f:
                item = json.loads(line)
                queries.append(item["query"])
                relevant.append(item["relevant"])

        return self.evaluate_retrieval(queries, relevant, top_k=top_k)
