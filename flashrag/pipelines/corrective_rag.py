"""
Corrective RAG (CRAG) pipeline.

Implements self-corrective retrieval-augmented generation, which evaluates
the relevance of retrieved documents and falls back to refined queries
or knowledge-augmented generation when initial retrieval is insufficient.

Reference: Yan et al., "Corrective Retrieval Augmented Generation" (2024)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from flashrag.embeddings.base import BaseEmbedding
from flashrag.generation.generator import RAGGenerator
from flashrag.pipelines.basic_rag import RAGResult
from flashrag.registry import PIPELINES
from flashrag.retrieval.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


@PIPELINES.register("corrective_rag")
class CorrectiveRAGPipeline:
    """
    Self-Corrective RAG (CRAG) pipeline.

    Three-phase retrieval process:
    1. **Retrieve**: Initial document retrieval
    2. **Grade**: Evaluate relevance of each retrieved document
    3. **Correct**: If relevance is low, refine query and re-retrieve

    Parameters
    ----------
    embedding_model : str or BaseEmbedding
        Embedding model for retrieval.
    generator_model : str
        LLM for generation and grading.
    top_k : int
        Number of documents to retrieve.
    relevance_threshold : float
        Score threshold for considering a document relevant.
    max_corrections : int
        Maximum number of correction iterations.
    min_relevant_docs : int
        Minimum relevant documents needed before generation.
    device : str
        Device for models.
    """

    def __init__(
        self,
        embedding_model: str | BaseEmbedding = "all-MiniLM-L6-v2",
        generator_model: str = "gpt2",
        top_k: int = 5,
        relevance_threshold: float = 0.35,
        max_corrections: int = 2,
        min_relevant_docs: int = 2,
        device: str = "cpu",
    ) -> None:
        if isinstance(embedding_model, str):
            from flashrag.embeddings.sentence_transformer import SentenceTransformerEmbedding

            self._embedder = SentenceTransformerEmbedding(embedding_model, device=device)
        else:
            self._embedder = embedding_model

        self._vector_store = VectorStore(dimension=self._embedder.dimension, metric="cosine")
        self._generator = RAGGenerator(model_name=generator_model, device=device)

        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        self.max_corrections = max_corrections
        self.min_relevant_docs = min_relevant_docs
        logger.info("CorrectiveRAGPipeline initialized")

    def index_documents(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        meta = metadata or [{}] * len(texts)
        vectors = self._embedder.encode(texts, show_progress=True)
        self._vector_store.add(vectors, texts, meta)
        return len(texts)

    def grade_documents(
        self, query: str, results: List[SearchResult]
    ) -> tuple[List[SearchResult], List[SearchResult]]:
        """
        Grade retrieved documents as relevant or irrelevant.

        Uses embedding similarity as a relevance proxy. Documents above
        the threshold are kept; those below are flagged for correction.
        """
        relevant: List[SearchResult] = []
        irrelevant: List[SearchResult] = []

        query_vec = self._embedder.encode([query])[0]

        for result in results:
            doc_vec = self._embedder.encode([result.text])[0]
            similarity = float(np.dot(query_vec, doc_vec))

            if similarity >= self.relevance_threshold:
                relevant.append(result)
            else:
                irrelevant.append(result)

        return relevant, irrelevant

    def refine_query(self, original_query: str, iteration: int) -> str:
        """
        Generate a refined query based on the original question.

        Applies query expansion strategies: adds specificity keywords,
        removes ambiguous terms, and reformulates.
        """
        expansions = [
            f"detailed explanation of {original_query}",
            f"specifically about {original_query} key concepts",
            f"{original_query} comprehensive overview",
        ]
        idx = iteration % len(expansions)
        return expansions[idx]

    def run(
        self,
        question: str,
        top_k: Optional[int] = None,
        **gen_kwargs: Any,
    ) -> RAGResult:
        """
        Execute the corrective RAG pipeline.

        1. Retrieve initial documents
        2. Grade relevance
        3. If insufficient, refine query and re-retrieve
        4. Generate answer from relevant documents
        """
        k = top_k or self.top_k
        all_relevant: List[SearchResult] = []
        seen_texts: set = set()
        corrections_made = 0

        query_vec = self._embedder.encode([question])[0]
        initial_results = self._vector_store.search(query_vec, top_k=k)

        relevant, irrelevant = self.grade_documents(question, initial_results)
        for r in relevant:
            if r.text[:100] not in seen_texts:
                seen_texts.add(r.text[:100])
                all_relevant.append(r)

        while (
            len(all_relevant) < self.min_relevant_docs
            and corrections_made < self.max_corrections
        ):
            corrections_made += 1
            refined_query = self.refine_query(question, corrections_made)
            logger.info(
                f"Correction {corrections_made}: refined query = '{refined_query[:80]}'"
            )

            refined_vec = self._embedder.encode([refined_query])[0]
            new_results = self._vector_store.search(refined_vec, top_k=k)
            new_relevant, _ = self.grade_documents(question, new_results)

            for r in new_relevant:
                if r.text[:100] not in seen_texts:
                    seen_texts.add(r.text[:100])
                    all_relevant.append(r)

        if not all_relevant:
            return RAGResult(
                answer=(
                    "I could not find sufficiently relevant information to answer "
                    "this question confidently."
                ),
                metadata={
                    "pipeline": "corrective_rag",
                    "corrections": corrections_made,
                    "status": "insufficient_evidence",
                },
            )

        gen_result = self._generator.generate(
            question=question,
            search_results=all_relevant[:k],
            **gen_kwargs,
        )

        return RAGResult(
            answer=gen_result.answer,
            contexts=[r.text for r in all_relevant],
            sources=[r.metadata for r in all_relevant],
            scores=[r.score for r in all_relevant],
            metadata={
                "pipeline": "corrective_rag",
                "corrections": corrections_made,
                "relevant_docs": len(all_relevant),
                "status": "success",
            },
        )
