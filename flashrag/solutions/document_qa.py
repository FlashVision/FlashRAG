"""
Document QA solution.

High-level API for question answering over a collection of documents.
Handles loading, chunking, indexing, and querying in a single interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from flashrag.data.preprocessor import Preprocessor
from flashrag.pipelines.basic_rag import BasicRAGPipeline, RAGResult

logger = logging.getLogger(__name__)


class DocumentQA:
    """
    Question answering over documents.

    A convenience wrapper that combines document loading, chunking,
    embedding, indexing, and RAG-based question answering.

    Parameters
    ----------
    embedding_model : str
        Embedding model name.
    generator_model : str
        Generator model name.
    chunk_size : int
        Document chunk size.
    chunk_overlap : int
        Overlap between chunks.
    top_k : int
        Number of contexts to retrieve.
    use_reranker : bool
        Whether to rerank results with a cross-encoder.
    device : str
        Device for models.

    Examples
    --------
    >>> qa = DocumentQA()
    >>> qa.add_documents(["paper.pdf", "notes.md"])
    >>> answer = qa.ask("What is the main finding?")
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        generator_model: str = "gpt2",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        use_reranker: bool = False,
        device: str = "cpu",
    ) -> None:
        self._pipeline = BasicRAGPipeline(
            embedding_model=embedding_model,
            generator_model=generator_model,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_reranker=use_reranker,
            device=device,
        )
        self._preprocessor = Preprocessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._indexed = False
        logger.info("DocumentQA initialized")

    def add_documents(self, paths: List[str | Path]) -> int:
        """Load and index documents from file paths."""
        count = self._pipeline.index_documents(paths=paths)
        self._indexed = True
        logger.info(f"Indexed {count} chunks from {len(paths)} files")
        return count

    def add_texts(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Index raw text strings."""
        count = self._pipeline.index_documents(texts=texts, metadata=metadata)
        self._indexed = True
        return count

    def ask(self, question: str, **kwargs: Any) -> str:
        """Ask a question and get a string answer."""
        if not self._indexed:
            raise RuntimeError("No documents indexed. Call add_documents() first.")
        return self._pipeline.query(question, **kwargs)

    def ask_detailed(self, question: str, **kwargs: Any) -> RAGResult:
        """Ask a question and get a full RAGResult with sources."""
        if not self._indexed:
            raise RuntimeError("No documents indexed. Call add_documents() first.")
        return self._pipeline.run(question, **kwargs)

    def save_index(self, path: str | Path) -> None:
        self._pipeline.save_index(path)

    def load_index(self, path: str | Path) -> None:
        self._pipeline.load_index(path)
        self._indexed = True
