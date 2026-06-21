"""
Basic RAG pipeline: Retrieve → Generate.

The standard retrieval-augmented generation flow: encode the query,
retrieve top-k documents from a vector store, format them into a prompt,
and generate an answer with an LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from flashrag.data.chunkers import Chunk
from flashrag.data.preprocessor import Preprocessor
from flashrag.embeddings.base import BaseEmbedding
from flashrag.generation.citation import CitationExtractor
from flashrag.generation.generator import GenerationResult, RAGGenerator
from flashrag.generation.prompt_templates import PromptTemplate, get_template
from flashrag.registry import PIPELINES
from flashrag.retrieval.reranker import CrossEncoderReranker
from flashrag.retrieval.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    answer: str
    contexts: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    citations: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@PIPELINES.register("basic_rag")
class BasicRAGPipeline:
    """
    Standard Retrieve → Generate pipeline.

    Parameters
    ----------
    embedding_model : str or BaseEmbedding
        Model for encoding queries and documents.
    generator_model : str
        HuggingFace model for answer generation.
    top_k : int
        Number of documents to retrieve.
    chunk_size : int
        Document chunk size in characters.
    chunk_overlap : int
        Overlap between chunks.
    prompt_template : str
        Name of the prompt template to use.
    use_reranker : bool
        Whether to apply cross-encoder reranking.
    reranker_model : str
        Model name for the cross-encoder reranker.
    reranker_top_k : int
        Number of results to keep after reranking.
    device : str
        Device for models.
    """

    def __init__(
        self,
        embedding_model: str | BaseEmbedding = "all-MiniLM-L6-v2",
        generator_model: str = "gpt2",
        top_k: int = 5,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        prompt_template: str = "default",
        use_reranker: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_top_k: int = 3,
        device: str = "cpu",
    ) -> None:
        if isinstance(embedding_model, str):
            from flashrag.embeddings.sentence_transformer import SentenceTransformerEmbedding

            self._embedder = SentenceTransformerEmbedding(embedding_model, device=device)
        else:
            self._embedder = embedding_model

        self._vector_store = VectorStore(dimension=self._embedder.dimension, metric="cosine")
        self._generator = RAGGenerator(
            model_name=generator_model,
            device=device,
            prompt_template=prompt_template,
        )
        self._preprocessor = Preprocessor(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self._citation_extractor = CitationExtractor()

        self.top_k = top_k
        self.use_reranker = use_reranker
        self._reranker: Optional[CrossEncoderReranker] = None
        if use_reranker:
            self._reranker = CrossEncoderReranker(model_name=reranker_model, device=device)
        self.reranker_top_k = reranker_top_k

        self._chunks: List[Chunk] = []
        logger.info("BasicRAGPipeline initialized")

    def index_documents(
        self,
        paths: Optional[List[str | Path]] = None,
        texts: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Load, chunk, embed, and index documents."""
        if paths:
            self._chunks = self._preprocessor.process_files(paths)
        elif texts:
            self._chunks = self._preprocessor.process_texts(texts, metadata)
        else:
            raise ValueError("Provide either 'paths' or 'texts'")

        chunk_texts = [c.text for c in self._chunks]
        chunk_meta = [c.metadata for c in self._chunks]

        logger.info(f"Encoding {len(chunk_texts)} chunks...")
        vectors = self._embedder.encode(chunk_texts, show_progress=True)
        self._vector_store.add(vectors, chunk_texts, chunk_meta)

        logger.info(f"Indexed {len(chunk_texts)} chunks")
        return len(chunk_texts)

    def run(
        self,
        question: str,
        top_k: Optional[int] = None,
        extract_citations: bool = True,
        **gen_kwargs: Any,
    ) -> RAGResult:
        """Execute the full RAG pipeline: retrieve → (rerank) → generate."""
        k = top_k or self.top_k

        query_vec = self._embedder.encode([question])[0]
        fetch_k = k * 3 if self.use_reranker else k
        results = self._vector_store.search(query_vec, top_k=fetch_k)

        if self.use_reranker and self._reranker:
            results = self._reranker.rerank(question, results, top_k=self.reranker_top_k)
        else:
            results = results[:k]

        gen_result = self._generator.generate(
            question=question,
            search_results=results,
            **gen_kwargs,
        )

        citation_report = None
        if extract_citations:
            citation_report = self._citation_extractor.extract(
                gen_result.answer,
                gen_result.contexts,
                gen_result.sources,
            )

        return RAGResult(
            answer=gen_result.answer,
            contexts=[r.text for r in results],
            sources=[r.metadata for r in results],
            scores=[r.score for r in results],
            citations=citation_report,
            metadata=gen_result.metadata,
        )

    def query(self, question: str, **kwargs: Any) -> str:
        """Convenience: run the pipeline and return just the answer string."""
        result = self.run(question, **kwargs)
        return result.answer

    def save_index(self, path: str | Path) -> None:
        self._vector_store.save(path)

    def load_index(self, path: str | Path) -> None:
        self._vector_store = VectorStore.load(path)
