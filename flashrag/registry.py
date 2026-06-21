"""
Pluggable component registry for FlashRAG.

Provides dictionaries that map string names → classes for embeddings,
retrievers, generators, pipelines, and rerankers.  Users and YAML configs
reference components by name; the registry resolves them at runtime.
"""

from __future__ import annotations

import importlib
from typing import Any


class _Registry:
    """A single named registry backed by a dict[str, type]."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._store: dict[str, type[Any]] = {}

    def register(self, name: str | None = None):
        """Decorator that registers a class under *name* (defaults to cls.__name__)."""

        def decorator(cls: type[Any]) -> type[Any]:
            key = name or cls.__name__
            if key in self._store:
                raise ValueError(
                    f"{self.name} registry already contains '{key}' → {self._store[key]}"
                )
            self._store[key] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[Any]:
        if name not in self._store:
            raise KeyError(
                f"'{name}' not found in {self.name} registry. Available: {list(self._store.keys())}"
            )
        return self._store[name]

    def list(self) -> list[str]:
        return list(self._store.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __repr__(self) -> str:
        return f"<Registry:{self.name} keys={self.list()}>"


EMBEDDINGS = _Registry("EMBEDDINGS")
RETRIEVERS = _Registry("RETRIEVERS")
GENERATORS = _Registry("GENERATORS")
PIPELINES = _Registry("PIPELINES")
RERANKERS = _Registry("RERANKERS")


def auto_register() -> None:
    """Force-import subpackages so @register decorators execute."""
    _modules = [
        "flashrag.embeddings.sentence_transformer",
        "flashrag.embeddings.openai_embeddings",
        "flashrag.embeddings.vision_embeddings",
        "flashrag.retrieval.vector_store",
        "flashrag.retrieval.bm25",
        "flashrag.retrieval.hybrid",
        "flashrag.retrieval.reranker",
        "flashrag.retrieval.colbert",
        "flashrag.retrieval.hyde",
        "flashrag.generation.generator",
        "flashrag.pipelines.basic_rag",
        "flashrag.pipelines.agentic_rag",
        "flashrag.pipelines.multimodal_rag",
        "flashrag.pipelines.corrective_rag",
        "flashrag.pipelines.graph_rag",
    ]
    for mod in _modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            pass
