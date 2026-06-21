"""
BM25 sparse retrieval.

Pure-Python implementation of Okapi BM25 with TF-IDF scoring.
No external dependencies beyond the standard library and numpy.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from flashrag.registry import RETRIEVERS
from flashrag.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)


def _tokenize(text: str, stop_words: set[str] | None = None) -> list[str]:
    """Lowercase, split on non-alphanumerics, remove stop words."""
    tokens = re.findall(r"\w+", text.lower())
    if stop_words:
        tokens = [t for t in tokens if t not in stop_words]
    return tokens


_DEFAULT_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "it", "of", "in", "to", "and", "or", "for",
    "on", "with", "as", "at", "by", "from", "that", "this", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "can",
    "not", "but", "if", "then", "than", "so", "no", "nor",
})


@RETRIEVERS.register("bm25")
class BM25Retriever:
    """
    Okapi BM25 sparse retriever.

    Parameters
    ----------
    k1 : float
        Term frequency saturation parameter (default 1.5).
    b : float
        Length normalization parameter (default 0.75).
    use_stop_words : bool
        Whether to filter common English stop words.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        use_stop_words: bool = True,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.stop_words = _DEFAULT_STOP_WORDS if use_stop_words else None

        self._documents: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_freqs: dict[str, int] = defaultdict(int)
        self._doc_lens: list[int] = []
        self._avg_dl: float = 0.0
        self._n_docs: int = 0

    @property
    def size(self) -> int:
        return self._n_docs

    def index(
        self,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build the BM25 index from a list of documents."""
        self._documents = list(documents)
        self._metadata = list(metadata) if metadata else [{}] * len(documents)
        self._n_docs = len(documents)

        self._doc_tokens = []
        self._doc_freqs = defaultdict(int)
        self._doc_lens = []

        for doc in documents:
            tokens = _tokenize(doc, self.stop_words)
            self._doc_tokens.append(tokens)
            self._doc_lens.append(len(tokens))
            seen: set[str] = set()
            for token in tokens:
                if token not in seen:
                    self._doc_freqs[token] += 1
                    seen.add(token)

        total_len = sum(self._doc_lens)
        self._avg_dl = total_len / self._n_docs if self._n_docs > 0 else 1.0
        logger.info(f"BM25 index built: {self._n_docs} documents, avg_dl={self._avg_dl:.1f}")

    def _bm25_score(self, query_tokens: list[str], doc_idx: int) -> float:
        doc_tokens = self._doc_tokens[doc_idx]
        doc_len = self._doc_lens[doc_idx]
        tf_counter = Counter(doc_tokens)

        score = 0.0
        for qt in query_tokens:
            if qt not in self._doc_freqs:
                continue
            df = self._doc_freqs[qt]
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)
            tf = tf_counter.get(qt, 0)
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_dl)
            )
            score += idf * tf_norm
        return score

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search the corpus and return top-k results ranked by BM25 score."""
        if self._n_docs == 0:
            return []

        query_tokens = _tokenize(query, self.stop_words)
        scores = np.array(
            [self._bm25_score(query_tokens, i) for i in range(self._n_docs)],
            dtype=np.float32,
        )

        k = min(top_k, self._n_docs)
        top_indices = np.argpartition(-scores, k)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: list[SearchResult] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append(
                SearchResult(
                    text=self._documents[idx],
                    score=float(scores[idx]),
                    metadata=self._metadata[idx],
                    vector_id=int(idx),
                )
            )
        return results

    def add_documents(
        self,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Incrementally add documents to an existing index."""
        new_meta = list(metadata) if metadata else [{}] * len(documents)
        all_docs = self._documents + list(documents)
        all_meta = self._metadata + new_meta
        self.index(all_docs, all_meta)
