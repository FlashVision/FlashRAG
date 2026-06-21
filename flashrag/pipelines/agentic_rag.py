"""
Agentic RAG pipeline with adaptive retrieval.

An agent-driven pipeline that can decompose complex queries into
sub-questions, perform multiple retrieval rounds, and synthesize
results. Implements query routing and iterative refinement.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from flashrag.embeddings.base import BaseEmbedding
from flashrag.generation.generator import RAGGenerator
from flashrag.pipelines.basic_rag import RAGResult
from flashrag.registry import PIPELINES
from flashrag.retrieval.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    action: str
    query: str
    results: list[SearchResult] = field(default_factory=list)
    reasoning: str = ""


@PIPELINES.register("agentic_rag")
class AgenticRAGPipeline:
    """
    Agent-driven adaptive RAG pipeline.

    Performs multi-step retrieval with query decomposition, relevance
    assessment, and iterative refinement until sufficient context is gathered.

    Parameters
    ----------
    embedding_model : str or BaseEmbedding
        Embedding model for dense retrieval.
    generator_model : str
        LLM for generation and agent reasoning.
    top_k : int
        Documents per retrieval step.
    max_steps : int
        Maximum number of retrieval iterations.
    relevance_threshold : float
        Minimum similarity score to consider a result relevant.
    device : str
        Device for models.
    """

    def __init__(
        self,
        embedding_model: str | BaseEmbedding = "all-MiniLM-L6-v2",
        generator_model: str = "gpt2",
        top_k: int = 5,
        max_steps: int = 3,
        relevance_threshold: float = 0.3,
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
        self.max_steps = max_steps
        self.relevance_threshold = relevance_threshold
        logger.info("AgenticRAGPipeline initialized")

    def index_documents(
        self,
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        meta = metadata or [{}] * len(texts)
        vectors = self._embedder.encode(texts, show_progress=True)
        self._vector_store.add(vectors, texts, meta)
        return len(texts)

    def decompose_query(self, question: str) -> list[str]:
        """
        Decompose a complex question into simpler sub-questions.

        Uses heuristic decomposition: splits compound questions on
        conjunctions and question words, then generates reformulations.
        """
        sub_questions = [question]

        compound_markers = [" and ", " also ", " additionally ", " furthermore "]
        for marker in compound_markers:
            if marker in question.lower():
                parts = re.split(re.escape(marker), question, flags=re.IGNORECASE)
                for part in parts:
                    part = part.strip().rstrip("?").strip() + "?"
                    if len(part) > 10:
                        sub_questions.append(part)

        compare_pattern = re.compile(
            r"(?:compare|difference|versus|vs\.?)\s+(.+?)\s+(?:and|with|vs\.?)\s+(.+)",
            re.IGNORECASE,
        )
        match = compare_pattern.search(question)
        if match:
            sub_questions.append(f"What is {match.group(1).strip()}?")
            sub_questions.append(f"What is {match.group(2).strip().rstrip('?')}?")

        seen = set()
        unique = []
        for q in sub_questions:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique.append(q)

        return unique

    def assess_relevance(
        self, query: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        """Filter results below the relevance threshold."""
        return [r for r in results if r.score >= self.relevance_threshold]

    def run(
        self,
        question: str,
        max_steps: int | None = None,
        **gen_kwargs: Any,
    ) -> RAGResult:
        """
        Execute the agentic RAG pipeline.

        1. Decompose the query into sub-questions
        2. For each sub-question, retrieve and assess relevance
        3. If insufficient context, try reformulated queries
        4. Synthesize all gathered context into a final answer
        """
        steps_limit = max_steps or self.max_steps
        all_results: list[SearchResult] = []
        agent_steps: list[AgentStep] = []
        seen_texts: set = set()

        sub_queries = self.decompose_query(question)

        for step_idx in range(steps_limit):
            if step_idx >= len(sub_queries):
                break

            query = sub_queries[step_idx]
            query_vec = self._embedder.encode([query])[0]
            results = self._vector_store.search(query_vec, top_k=self.top_k)

            relevant = self.assess_relevance(query, results)

            step = AgentStep(
                action=f"retrieve_step_{step_idx + 1}",
                query=query,
                results=relevant,
                reasoning=f"Retrieved {len(results)} results, {len(relevant)} relevant",
            )
            agent_steps.append(step)

            for r in relevant:
                if r.text[:100] not in seen_texts:
                    seen_texts.add(r.text[:100])
                    all_results.append(r)

        if not all_results:
            return RAGResult(
                answer="I could not find relevant information to answer this question.",
                metadata={"agent_steps": len(agent_steps), "pipeline": "agentic"},
            )

        gen_result = self._generator.generate(
            question=question,
            search_results=all_results[:self.top_k * 2],
            **gen_kwargs,
        )

        return RAGResult(
            answer=gen_result.answer,
            contexts=[r.text for r in all_results],
            sources=[r.metadata for r in all_results],
            scores=[r.score for r in all_results],
            metadata={
                "agent_steps": len(agent_steps),
                "sub_queries": sub_queries,
                "pipeline": "agentic",
            },
        )
