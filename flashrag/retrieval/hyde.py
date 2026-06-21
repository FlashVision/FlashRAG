"""
HyDE – Hypothetical Document Embeddings retrieval.

Before searching, HyDE generates one or more hypothetical answers to the
query, embeds those answers, and uses the resulting vectors to retrieve
real documents.  This bridges the query-document vocabulary gap and
improves recall on knowledge-intensive tasks.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from flashrag.registry import RETRIEVERS
from flashrag.retrieval.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

_FALLBACK_TEMPLATE = (
    "A possible answer to the question \"{query}\" would discuss the "
    "following relevant information and key concepts related to the topic."
)

_LLM_PROMPT_TEMPLATE = (
    "Please write a short passage that directly answers the following "
    "question. Write as if it were a paragraph from a Wikipedia article.\n\n"
    "Question: {query}\n\nPassage:"
)


@RETRIEVERS.register("hyde")
class HyDERetriever:
    """
    HyDE (Hypothetical Document Embeddings) retriever.

    Generates hypothetical documents for a query, embeds them, and
    retrieves real documents using the hypothetical embedding.
    Falls back to a template-based expansion when no LLM is available.

    Parameters
    ----------
    base_retriever : object, optional
        Any retriever that exposes ``search(query_vector, top_k)`` and
        ``add(vectors, documents, metadata)`` (e.g. a ``VectorStore``).
        If *None*, an internal ``VectorStore`` is created.
    generator_model : str
        Model name passed to an LLM for hypothesis generation.
    embedding_model : object, optional
        Embedding model with an ``encode(texts)`` method.  When *None*,
        a ``SentenceTransformerEmbedding`` is lazily loaded.
    num_hypothetical : int
        Number of hypothetical documents to generate per query.
    dimension : int
        Embedding dimension (used when creating the internal vector store).
    """

    def __init__(
        self,
        base_retriever: Any = None,
        generator_model: str = "gpt-3.5-turbo",
        embedding_model: Any = None,
        num_hypothetical: int = 1,
        dimension: int = 384,
    ) -> None:
        self.generator_model = generator_model
        self.num_hypothetical = num_hypothetical
        self.dimension = dimension

        self._embedding_model = embedding_model
        self._base_retriever = base_retriever or VectorStore(
            dimension=dimension, metric="cosine",
        )
        self._llm_available: bool | None = None
        self._llm: Any = None

        logger.info(
            "HyDERetriever created (generator=%s, n_hypo=%d, dim=%d)",
            generator_model, num_hypothetical, dimension,
        )

    def _ensure_embedding_model(self) -> None:
        """Lazy-load the embedding model if none was provided."""
        if self._embedding_model is not None:
            return

        from flashrag.embeddings.sentence_transformer import (
            SentenceTransformerEmbedding,
        )

        self._embedding_model = SentenceTransformerEmbedding(
            "all-MiniLM-L6-v2",
        )
        self.dimension = self._embedding_model.dimension
        logger.info("HyDE loaded default embedding model (all-MiniLM-L6-v2)")

    def _try_init_llm(self) -> bool:
        """Attempt to initialise an OpenAI-compatible LLM client."""
        if self._llm_available is not None:
            return self._llm_available

        try:
            import openai  # noqa: F811
            self._llm = openai.OpenAI()
            self._llm_available = True
            logger.info("HyDE LLM client initialised (%s)", self.generator_model)
        except Exception:
            self._llm_available = False
            logger.info(
                "No LLM available – HyDE will use template-based expansion"
            )
        return self._llm_available

    def _generate_hypothetical(self, query: str) -> list[str]:
        """
        Generate hypothetical answer documents for *query*.

        If an OpenAI-compatible LLM is configured and reachable, the
        model is used.  Otherwise a deterministic template expansion
        produces the hypothetical text.

        Parameters
        ----------
        query : str
            The user's search query.

        Returns
        -------
        list[str]
            One or more hypothetical answer passages.
        """
        if self._try_init_llm():
            return self._generate_with_llm(query)
        return self._generate_with_template(query)

    def _generate_with_llm(self, query: str) -> list[str]:
        """Call the LLM to produce hypothetical passages."""
        prompt = _LLM_PROMPT_TEMPLATE.format(query=query)
        hypotheticals: list[str] = []
        for _ in range(self.num_hypothetical):
            try:
                response = self._llm.chat.completions.create(
                    model=self.generator_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                    temperature=0.7,
                )
                text = response.choices[0].message.content.strip()
                if text:
                    hypotheticals.append(text)
            except Exception:
                logger.warning(
                    "LLM call failed, falling back to template expansion"
                )
                hypotheticals.append(
                    _FALLBACK_TEMPLATE.format(query=query),
                )
        return hypotheticals or [_FALLBACK_TEMPLATE.format(query=query)]

    def _generate_with_template(self, query: str) -> list[str]:
        """Deterministic template expansion for when no LLM is available."""
        templates = [
            _FALLBACK_TEMPLATE,
            (
                "The answer to \"{query}\" involves understanding the "
                "underlying principles, definitions, and relationships "
                "between the key entities mentioned in the question."
            ),
            (
                "To address \"{query}\", one should consider the relevant "
                "facts, prior research, and established knowledge in "
                "this domain."
            ),
        ]
        return [
            t.format(query=query)
            for t in templates[:self.num_hypothetical]
        ]

    @property
    def size(self) -> int:
        """Number of indexed documents in the underlying store."""
        return self._base_retriever.size

    def index(
        self,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Index documents into the underlying retriever.

        Each document is embedded and stored in the base retriever's
        vector index.

        Parameters
        ----------
        documents : list[str]
            Corpus of documents to index.
        metadata : list[dict], optional
            Per-document metadata dicts.
        """
        self._ensure_embedding_model()
        meta = list(metadata) if metadata else [{}] * len(documents)

        logger.info("HyDE: encoding %d documents ...", len(documents))
        vectors = self._embedding_model.encode(documents, show_progress=True)
        self._base_retriever.add(vectors, documents, meta)
        logger.info("HyDE index built: %d documents", self.size)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Search by generating a hypothetical answer and using its embedding.

        Parameters
        ----------
        query : str
            Natural-language query.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[SearchResult]
            Ranked retrieval results.
        """
        if self.size == 0:
            return []

        self._ensure_embedding_model()

        hypotheticals = self._generate_hypothetical(query)
        logger.debug(
            "Generated %d hypothetical doc(s) for query", len(hypotheticals),
        )

        hypo_vectors = self._embedding_model.encode(hypotheticals)
        query_vector = np.mean(hypo_vectors, axis=0).astype(np.float32)

        return self._base_retriever.search(query_vector, top_k=top_k)
