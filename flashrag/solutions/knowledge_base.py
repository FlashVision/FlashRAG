"""
Knowledge Base solution.

A persistent knowledge base that can be incrementally updated with
new documents and queried for information retrieval and QA.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from flashrag.data.preprocessor import Preprocessor
from flashrag.generation.generator import RAGGenerator
from flashrag.retrieval.hybrid import HybridSearch
from flashrag.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Persistent knowledge base with hybrid retrieval.

    Combines dense and sparse search for robust retrieval across
    a growing document collection. Supports save/load for persistence.

    Parameters
    ----------
    embedding_model : str
        Embedding model name.
    generator_model : str
        Generator model for QA (optional).
    hybrid_alpha : float
        Weight for dense vs sparse search (1.0 = all dense).
    persist_dir : str, optional
        Directory for persisting the index.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        generator_model: str | None = None,
        hybrid_alpha: float = 0.5,
        persist_dir: str | Path | None = None,
    ) -> None:
        self._hybrid = HybridSearch(
            embedding_model=embedding_model,
            alpha=hybrid_alpha,
        )
        self._generator: RAGGenerator | None = None
        if generator_model:
            self._generator = RAGGenerator(model_name=generator_model)

        self._preprocessor = Preprocessor()
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._doc_count = 0

        if self._persist_dir and (self._persist_dir / "kb_state.json").exists():
            self._load_state()

        logger.info("KnowledgeBase initialized")

    def add_documents(
        self,
        paths: list[str | Path] | None = None,
        texts: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """Add documents to the knowledge base."""
        if paths:
            chunks = self._preprocessor.process_files(paths)
            chunk_texts = [c.text for c in chunks]
            chunk_meta = [c.metadata for c in chunks]
        elif texts:
            chunk_texts = list(texts)
            chunk_meta = list(metadata) if metadata else [{}] * len(texts)
        else:
            raise ValueError("Provide either 'paths' or 'texts'")

        if not self._doc_count:
            self._hybrid.index(chunk_texts, chunk_meta)
        else:
            self._hybrid.add_documents(chunk_texts, chunk_meta)

        self._doc_count += len(chunk_texts)

        if self._persist_dir:
            self._save_state()

        logger.info(f"Added {len(chunk_texts)} chunks (total: {self._doc_count})")
        return len(chunk_texts)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search the knowledge base."""
        return self._hybrid.search(query, top_k=top_k)

    def ask(self, question: str, top_k: int = 5) -> str:
        """Ask a question against the knowledge base."""
        if not self._generator:
            raise RuntimeError(
                "No generator model configured. Initialize with generator_model= to enable QA."
            )

        results = self.search(question, top_k=top_k)
        gen_result = self._generator.generate(
            question=question,
            search_results=results,
        )
        return gen_result.answer

    @property
    def size(self) -> int:
        return self._doc_count

    def _save_state(self) -> None:
        if not self._persist_dir:
            return
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        state_path = self._persist_dir / "kb_state.json"
        with open(state_path, "w") as f:
            json.dump({"doc_count": self._doc_count}, f)

    def _load_state(self) -> None:
        if not self._persist_dir:
            return
        state_path = self._persist_dir / "kb_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            self._doc_count = state.get("doc_count", 0)
