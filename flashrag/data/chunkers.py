"""
Text chunking strategies for splitting documents into retrieval-sized pieces.

Provides recursive, sentence-level, and fixed-size chunkers that preserve
metadata and optionally overlap for context continuity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(idx={self.chunk_index}, len={len(self.text)}, '{preview}…')"


class RecursiveChunker:
    """
    Recursively split text using a hierarchy of separators.

    Tries double-newline first, then single newline, then sentence boundary,
    then word boundary. This produces semantically coherent chunks.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        metadata = metadata or {}
        raw_chunks = self._split_recursive(text, self.separators)
        merged = self._merge_with_overlap(raw_chunks)
        return [
            Chunk(text=t, metadata={**metadata, "chunk_index": i}, chunk_index=i)
            for i, t in enumerate(merged)
        ]

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else [""]

        if not sep:
            return [text[i: i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(sep)
        result: list[str] = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(part) > self.chunk_size:
                    result.extend(self._split_recursive(part, remaining_seps))
                    current = ""
                else:
                    current = part

        if current:
            result.append(current)

        return result

    def _merge_with_overlap(self, chunks: list[str]) -> list[str]:
        if not chunks or self.chunk_overlap <= 0:
            return chunks

        result: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
            merged = overlap_text + chunks[i]
            result.append(merged)
        return result


class SentenceChunker:
    """
    Split text into chunks at sentence boundaries.

    Groups sentences until the chunk reaches ``chunk_size`` characters,
    then starts a new chunk with optional sentence overlap.
    """

    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        chunk_size: int = 512,
        sentence_overlap: int = 1,
    ) -> None:
        self.chunk_size = chunk_size
        self.sentence_overlap = sentence_overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        metadata = metadata or {}
        sentences = self._SENTENCE_RE.split(text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) > self.chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={**metadata, "chunk_index": len(chunks)},
                        chunk_index=len(chunks),
                    )
                )
                overlap = current_sentences[-self.sentence_overlap:]
                current_sentences = list(overlap)
                current_len = sum(len(s) for s in current_sentences) + len(current_sentences) - 1
            current_sentences.append(sent)
            current_len += len(sent) + 1

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={**metadata, "chunk_index": len(chunks)},
                    chunk_index=len(chunks),
                )
            )

        return chunks


class FixedChunker:
    """Split text into fixed-size character chunks with overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        metadata = metadata or {}
        step = max(self.chunk_size - self.chunk_overlap, 1)
        chunks: list[Chunk] = []
        for i in range(0, len(text), step):
            segment = text[i: i + self.chunk_size]
            if segment.strip():
                chunks.append(
                    Chunk(
                        text=segment,
                        metadata={**metadata, "chunk_index": len(chunks)},
                        chunk_index=len(chunks),
                    )
                )
        return chunks
