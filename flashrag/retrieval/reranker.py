"""
Cross-encoder reranking for improving retrieval precision.

Takes initial retrieval results and re-scores them using a cross-encoder
model that jointly processes the query and each candidate document.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from flashrag.registry import RERANKERS
from flashrag.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)


@RERANKERS.register("cross_encoder")
class CrossEncoderReranker:
    """
    Rerank retrieval results using a cross-encoder model.

    Cross-encoders process ``(query, document)`` pairs jointly and produce
    a relevance score, yielding higher precision than bi-encoders at the
    cost of latency (can't be pre-computed).

    Parameters
    ----------
    model_name : str
        HuggingFace cross-encoder model, e.g.
        ``"cross-encoder/ms-marco-MiniLM-L-6-v2"``.
    device : str
        ``"cpu"`` or ``"cuda"``.
    max_length : int
        Maximum token length for cross-encoder input.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self._model.eval()
        logger.info(f"CrossEncoderReranker loaded: {model_name} (device={device})")

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Re-score and re-sort retrieval results by cross-encoder relevance.

        Parameters
        ----------
        query : str
            The original search query.
        results : list of SearchResult
            Initial retrieval results to rerank.
        top_k : int, optional
            Return only the top-k reranked results. Default: return all.
        """
        if not results:
            return []

        pairs = [(query, r.text) for r in results]
        scores = self._score_pairs(pairs)

        reranked = []
        for result, score in zip(results, scores):
            reranked.append(
                SearchResult(
                    text=result.text,
                    score=float(score),
                    metadata={**result.metadata, "original_score": result.score},
                    vector_id=result.vector_id,
                )
            )

        reranked.sort(key=lambda r: r.score, reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

    def _score_pairs(self, pairs: list[tuple]) -> np.ndarray:
        """Score a batch of (query, document) pairs."""
        batch_size = 32
        all_scores: list[float] = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            queries, docs = zip(*batch)

            inputs = self._tokenizer(
                list(queries),
                list(docs),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                if logits.shape[-1] == 1:
                    scores = logits.squeeze(-1)
                else:
                    scores = logits[:, 1]
                all_scores.extend(scores.cpu().tolist())

        return np.array(all_scores, dtype=np.float32)

    def score(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        return float(self._score_pairs([(query, document)])[0])
