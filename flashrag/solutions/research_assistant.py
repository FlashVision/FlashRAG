"""
Research Assistant solution.

An advanced RAG solution that uses agentic retrieval with query
decomposition for complex research questions across large document
collections.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from flashrag.data.preprocessor import Preprocessor
from flashrag.generation.citation import CitationExtractor
from flashrag.pipelines.agentic_rag import AgenticRAGPipeline

logger = logging.getLogger(__name__)


class ResearchAssistant:
    """
    AI research assistant for complex multi-source queries.

    Uses agentic RAG for adaptive retrieval with query decomposition,
    multi-hop reasoning, and citation extraction.

    Parameters
    ----------
    embedding_model : str
        Embedding model for retrieval.
    generator_model : str
        LLM for answer generation.
    top_k : int
        Documents per retrieval step.
    max_steps : int
        Maximum agent retrieval steps.
    device : str
        Device for models.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        generator_model: str = "gpt2",
        top_k: int = 5,
        max_steps: int = 3,
        device: str = "cpu",
    ) -> None:
        self._pipeline = AgenticRAGPipeline(
            embedding_model=embedding_model,
            generator_model=generator_model,
            top_k=top_k,
            max_steps=max_steps,
            device=device,
        )
        self._preprocessor = Preprocessor()
        self._citation_extractor = CitationExtractor()
        self._indexed = False
        logger.info("ResearchAssistant initialized")

    def add_documents(
        self,
        paths: list[str | Path] | None = None,
        texts: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """Add research documents."""
        if paths:
            chunks = self._preprocessor.process_files(paths)
            chunk_texts = [c.text for c in chunks]
            chunk_meta = [c.metadata for c in chunks]
        elif texts:
            chunk_texts = list(texts)
            chunk_meta = list(metadata) if metadata else [{}] * len(texts)
        else:
            raise ValueError("Provide either 'paths' or 'texts'")

        count = self._pipeline.index_documents(chunk_texts, chunk_meta)
        self._indexed = True
        return count

    def research(
        self,
        question: str,
        extract_citations: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Conduct research on a question.

        Returns a dict with the answer, citations, sources, and
        agent metadata (sub-queries used, steps taken, etc.).
        """
        if not self._indexed:
            raise RuntimeError("No documents indexed. Call add_documents() first.")

        result = self._pipeline.run(question, **kwargs)

        response: dict[str, Any] = {
            "answer": result.answer,
            "sources": result.sources,
            "num_sources": len(result.sources),
            "metadata": result.metadata,
        }

        if extract_citations and result.contexts:
            citation_report = self._citation_extractor.extract(
                result.answer, result.contexts
            )
            response["citations"] = {
                "cited_sources": citation_report.cited_sources,
                "uncited_sources": citation_report.uncited_sources,
                "attribution_score": citation_report.attribution_score,
                "num_citations": len(citation_report.citations),
            }

        return response

    def ask(self, question: str, **kwargs: Any) -> str:
        """Simple interface: ask a question, get a string answer."""
        result = self.research(question, extract_citations=False, **kwargs)
        return result["answer"]
