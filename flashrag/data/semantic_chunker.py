"""
Semantic chunking with embedding-based breakpoint detection.

Splits text at natural topic boundaries by measuring embedding similarity
between consecutive sentences. Where similarity drops below a percentile
threshold, a chunk boundary is inserted.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from flashrag.data.chunkers import Chunk

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Chunk text by detecting semantic breakpoints between consecutive sentences.

    Computes embeddings (or TF-IDF vectors) for each sentence, measures cosine
    similarity between neighbours, and inserts a chunk boundary wherever
    similarity drops below the configured percentile threshold.

    Parameters
    ----------
    embedding_model : object or None
        An embedding model with an ``encode(texts) -> np.ndarray`` method.
        If *None*, falls back to a lightweight TF-IDF approach.
    similarity_threshold : float
        Absolute similarity threshold below which a breakpoint is inserted
        (used only when ``breakpoint_percentile`` is *None*).
    min_chunk_size : int
        Minimum chunk size in characters. Smaller chunks are merged with
        their neighbours.
    max_chunk_size : int
        Maximum chunk size in characters. Larger chunks are split at
        sentence boundaries.
    breakpoint_percentile : int or None
        Percentile of the similarity distribution used to determine the
        dynamic breakpoint threshold. Set to *None* to use the absolute
        ``similarity_threshold`` instead.
    """

    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        embedding_model: Any | None = None,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        breakpoint_percentile: int | None = 90,
    ) -> None:
        self._embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.breakpoint_percentile = breakpoint_percentile

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """
        Split *text* into semantically coherent chunks.

        Parameters
        ----------
        text : str
            The input document text.
        metadata : dict or None
            Optional metadata to attach to each produced chunk.

        Returns
        -------
        list[Chunk]
            Ordered list of chunks with metadata and indices.
        """
        metadata = metadata or {}

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        if len(sentences) == 1:
            return [Chunk(text=sentences[0], metadata={**metadata, "chunk_index": 0},
                          chunk_index=0)]

        similarities = self._compute_similarities(sentences)
        breakpoints = self._find_breakpoints(similarities)

        raw_chunks = self._group_by_breakpoints(sentences, breakpoints)
        raw_chunks = self._merge_small_chunks(raw_chunks, self.min_chunk_size)
        raw_chunks = self._split_large_chunks(raw_chunks, self.max_chunk_size)

        return [
            Chunk(
                text=t,
                metadata={**metadata, "chunk_index": i},
                chunk_index=i,
            )
            for i, t in enumerate(raw_chunks)
            if t.strip()
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> list[str]:
        """Split *text* into sentences using punctuation boundaries."""
        parts = self._SENTENCE_RE.split(text.strip())
        return [s.strip() for s in parts if s.strip()]

    def _compute_similarities(self, sentences: list[str]) -> np.ndarray:
        """
        Compute cosine similarity between consecutive sentence embeddings.

        Returns an array of length ``len(sentences) - 1``.
        """
        embeddings = self._encode_sentences(sentences)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = embeddings / norms

        similarities = np.array([
            float(np.dot(normed[i], normed[i + 1]))
            for i in range(len(normed) - 1)
        ])
        return similarities

    def _find_breakpoints(self, similarities: np.ndarray) -> list[int]:
        """
        Identify indices where a chunk boundary should be inserted.

        Parameters
        ----------
        similarities : np.ndarray
            Cosine similarities between consecutive sentences.

        Returns
        -------
        list[int]
            Sorted indices (into the similarities array) marking breakpoints.
            A breakpoint at index *i* means a split occurs *after* sentence *i*.
        """
        if len(similarities) == 0:
            return []

        if self.breakpoint_percentile is not None:
            threshold = float(
                np.percentile(similarities, 100 - self.breakpoint_percentile)
            )
        else:
            threshold = self.similarity_threshold

        breakpoints = [
            i for i, sim in enumerate(similarities) if sim < threshold
        ]
        return sorted(breakpoints)

    def _merge_small_chunks(
        self, chunks: list[str], min_size: int
    ) -> list[str]:
        """Merge consecutive chunks that are below *min_size* characters."""
        if not chunks:
            return chunks

        merged: list[str] = [chunks[0]]
        for chunk in chunks[1:]:
            if len(merged[-1]) < min_size:
                merged[-1] = merged[-1] + " " + chunk
            else:
                merged.append(chunk)

        if len(merged) > 1 and len(merged[-1]) < min_size:
            merged[-2] = merged[-2] + " " + merged[-1]
            merged.pop()

        return merged

    def _split_large_chunks(
        self, chunks: list[str], max_size: int
    ) -> list[str]:
        """Split chunks exceeding *max_size* at the nearest sentence boundary."""
        result: list[str] = []
        for chunk in chunks:
            if len(chunk) <= max_size:
                result.append(chunk)
                continue

            sentences = self._split_sentences(chunk)
            current = ""
            for sent in sentences:
                candidate = (current + " " + sent).strip() if current else sent
                if len(candidate) > max_size and current:
                    result.append(current)
                    current = sent
                else:
                    current = candidate

            if current:
                result.append(current)

        return result

    # ------------------------------------------------------------------
    # Embedding logic
    # ------------------------------------------------------------------

    def _encode_sentences(self, sentences: list[str]) -> np.ndarray:
        """Encode sentences into vector representations."""
        if self._embedding_model is not None:
            return np.asarray(self._embedding_model.encode(sentences))

        return self._tfidf_encode(sentences)

    def _tfidf_encode(self, sentences: list[str]) -> np.ndarray:
        """Lightweight TF-IDF fallback when no embedding model is provided."""
        tokenized = [set(s.lower().split()) for s in sentences]
        vocab: dict[str, int] = {}
        for tokens in tokenized:
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)

        if not vocab:
            return np.zeros((len(sentences), 1))

        vectors = np.zeros((len(sentences), len(vocab)))
        doc_freq = np.zeros(len(vocab))

        for tokens in tokenized:
            for tok in tokens:
                doc_freq[vocab[tok]] += 1

        idf = np.log((len(sentences) + 1) / (doc_freq + 1)) + 1

        for i, tokens in enumerate(tokenized):
            for tok in tokens:
                vectors[i, vocab[tok]] = 1.0
            vectors[i] *= idf

        return vectors

    # ------------------------------------------------------------------
    # Grouping helper
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_breakpoints(
        sentences: list[str], breakpoints: list[int]
    ) -> list[str]:
        """Group sentences into chunks separated by breakpoint indices."""
        chunks: list[str] = []
        prev = 0
        for bp in breakpoints:
            chunk_text = " ".join(sentences[prev: bp + 1])
            if chunk_text.strip():
                chunks.append(chunk_text)
            prev = bp + 1

        remaining = " ".join(sentences[prev:])
        if remaining.strip():
            chunks.append(remaining)

        return chunks
