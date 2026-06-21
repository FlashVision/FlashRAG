"""
Document preprocessing pipeline.

Combines loading and chunking into a single ``Preprocessor`` that takes
raw file paths and returns ready-to-embed ``Chunk`` objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from flashrag.data.chunkers import Chunk, FixedChunker, RecursiveChunker, SentenceChunker
from flashrag.data.loaders import Document, DocumentLoader

logger = logging.getLogger(__name__)

_CHUNKER_MAP = {
    "recursive": RecursiveChunker,
    "sentence": SentenceChunker,
    "fixed": FixedChunker,
}


class Preprocessor:
    """
    End-to-end document preprocessor: load → clean → chunk.

    Parameters
    ----------
    chunk_size : int
        Target chunk size in characters.
    chunk_overlap : int
        Overlap between consecutive chunks.
    chunk_strategy : str
        One of ``"recursive"``, ``"sentence"``, ``"fixed"``.
    min_chunk_length : int
        Discard chunks shorter than this.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        chunk_strategy: str = "recursive",
        min_chunk_length: int = 20,
    ) -> None:
        self.loader = DocumentLoader()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

        if chunk_strategy not in _CHUNKER_MAP:
            raise ValueError(
                f"Unknown chunk strategy '{chunk_strategy}'. "
                f"Choose from: {list(_CHUNKER_MAP.keys())}"
            )
        chunker_cls = _CHUNKER_MAP[chunk_strategy]
        if chunk_strategy == "sentence":
            self.chunker = chunker_cls(chunk_size=chunk_size)
        else:
            self.chunker = chunker_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def process_files(self, paths: List[str | Path]) -> List[Chunk]:
        """Load and chunk multiple files."""
        all_chunks: List[Chunk] = []
        for path in paths:
            path = Path(path)
            if not path.exists():
                logger.warning(f"Path does not exist, skipping: {path}")
                continue
            docs = self.loader.load(path)
            for doc in docs:
                chunks = self._process_document(doc)
                all_chunks.extend(chunks)
        logger.info(f"Preprocessed {len(paths)} paths → {len(all_chunks)} chunks")
        return all_chunks

    def process_texts(
        self,
        texts: List[str],
        metadata: Optional[List[dict]] = None,
    ) -> List[Chunk]:
        """Chunk raw text strings directly."""
        metadata = metadata or [{}] * len(texts)
        all_chunks: List[Chunk] = []
        for text, meta in zip(texts, metadata):
            doc = Document(text=text, metadata=meta)
            all_chunks.extend(self._process_document(doc))
        return all_chunks

    def _process_document(self, doc: Document) -> List[Chunk]:
        cleaned = self._clean_text(doc.text)
        if len(cleaned) < self.min_chunk_length:
            return []
        chunks = self.chunker.chunk(cleaned, metadata=doc.metadata)
        return [c for c in chunks if len(c.text.strip()) >= self.min_chunk_length]

    @staticmethod
    def _clean_text(text: str) -> str:
        import re

        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
