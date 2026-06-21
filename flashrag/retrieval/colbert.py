"""
ColBERT late-interaction retrieval.

Implements the ColBERT scoring model where queries and documents are
encoded into per-token embeddings.  Relevance is computed via MaxSim:
for each query token, take the maximum cosine similarity over all
document tokens, then sum across query tokens.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from flashrag.registry import RETRIEVERS
from flashrag.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)


@RETRIEVERS.register("colbert")
class ColBERTRetriever:
    """
    ColBERT late-interaction retriever.

    Encodes queries and documents into per-token embeddings and scores
    them with MaxSim.  Uses a ``transformers`` ColBERT checkpoint for
    encoding and stores all document-token matrices in memory.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the ColBERT checkpoint.
    dimension : int
        Output embedding dimension per token.
    device : str
        Torch device (``"cpu"``, ``"cuda"``, etc.).
    max_doc_length : int
        Maximum token length for documents.
    max_query_length : int
        Maximum token length for queries.
    index_batch_size : int
        Number of documents encoded per forward pass during indexing.
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        dimension: int = 128,
        device: str = "cpu",
        max_doc_length: int = 512,
        max_query_length: int = 32,
        index_batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.device = device
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        self.index_batch_size = index_batch_size

        self._tokenizer: Any = None
        self._model: Any = None

        self._doc_embeddings: list[np.ndarray] = []
        self._documents: list[str] = []
        self._metadata: list[dict[str, Any]] = []

        logger.info(
            "ColBERTRetriever created (model=%s, dim=%d, device=%s)",
            model_name,
            dimension,
            device,
        )

    def _load_model(self) -> None:
        """Lazy-load the transformer model and tokenizer.

        Falls back to a lightweight hash-based encoder when the HuggingFace
        model is unavailable (no network, missing ``transformers``, etc.).
        """
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            logger.info("Loading ColBERT model %s ...", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            self._torch = torch
            self._use_fallback = False
            logger.info("ColBERT model loaded on %s", self.device)
        except Exception:
            logger.warning(
                "ColBERT transformer model unavailable, using lightweight "
                "hash-based fallback encoder."
            )
            self._model = "fallback"
            self._use_fallback = True

    def _encode_tokens(
        self,
        text: str,
        max_length: int,
    ) -> np.ndarray:
        """
        Encode *text* into per-token embeddings.

        Returns
        -------
        np.ndarray
            Shape ``(num_tokens, dimension)`` with L2-normalised vectors.
        """
        self._load_model()

        if self._use_fallback:
            return self._encode_tokens_fallback(text, max_length)

        torch = self._torch

        encoding = self._tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**encoding)

        token_embeds = outputs.last_hidden_state[0].cpu().numpy()
        attention_mask = encoding["attention_mask"][0].cpu().numpy()

        token_embeds = token_embeds[attention_mask.astype(bool)]

        if token_embeds.shape[1] != self.dimension:
            token_embeds = token_embeds[:, : self.dimension]

        norms = np.linalg.norm(token_embeds, axis=1, keepdims=True)
        token_embeds = token_embeds / np.maximum(norms, 1e-12)
        return token_embeds.astype(np.float32)

    def _encode_tokens_fallback(
        self,
        text: str,
        max_length: int,
    ) -> np.ndarray:
        """Hash-based per-token encoding fallback when no model is available."""
        import hashlib

        tokens = text.lower().split()[:max_length]
        if not tokens:
            tokens = ["[empty]"]

        embeddings = np.zeros((len(tokens), self.dimension), dtype=np.float32)
        for i, token in enumerate(tokens):
            digest = hashlib.sha256(token.encode()).digest()
            rng = np.random.RandomState(int.from_bytes(digest[:4], "big"))
            vec = rng.randn(self.dimension).astype(np.float32)
            norm = np.linalg.norm(vec)
            embeddings[i] = vec / max(norm, 1e-12)
        return embeddings

    def _encode_query(self, text: str) -> np.ndarray:
        """
        Encode a query into per-token embeddings.

        Parameters
        ----------
        text : str
            Raw query string.

        Returns
        -------
        np.ndarray
            Shape ``(num_query_tokens, dimension)``.
        """
        return self._encode_tokens(text, self.max_query_length)

    def _encode_document(self, text: str) -> np.ndarray:
        """
        Encode a document into per-token embeddings.

        Parameters
        ----------
        text : str
            Raw document string.

        Returns
        -------
        np.ndarray
            Shape ``(num_doc_tokens, dimension)``.
        """
        return self._encode_tokens(text, self.max_doc_length)

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return len(self._documents)

    def index(
        self,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Encode and store per-token embeddings for every document.

        Parameters
        ----------
        documents : list[str]
            Corpus of documents to index.
        metadata : list[dict], optional
            Per-document metadata dicts.
        """
        self._load_model()
        self._documents = list(documents)
        self._metadata = list(metadata) if metadata else [{}] * len(documents)
        self._doc_embeddings = []

        n = len(documents)
        for start in range(0, n, self.index_batch_size):
            end = min(start + self.index_batch_size, n)
            batch = documents[start:end]
            for doc in batch:
                emb = self._encode_document(doc)
                self._doc_embeddings.append(emb)
            logger.debug(
                "Indexed documents %d–%d / %d",
                start,
                end - 1,
                n,
            )

        logger.info("ColBERT index built: %d documents", self.size)

    @staticmethod
    def _maxsim_score(
        query_embeddings: np.ndarray,
        doc_embeddings: np.ndarray,
    ) -> float:
        """
        Compute the MaxSim relevance score between a query and a document.

        For each query token, find the maximum cosine similarity over all
        document tokens, then sum across query tokens.

        Parameters
        ----------
        query_embeddings : np.ndarray
            Shape ``(Q, D)`` – per-token query vectors (L2-normalised).
        doc_embeddings : np.ndarray
            Shape ``(T, D)`` – per-token document vectors (L2-normalised).

        Returns
        -------
        float
            The MaxSim score (higher is better).
        """
        sim_matrix = query_embeddings @ doc_embeddings.T  # (Q, T)
        max_per_query_token = sim_matrix.max(axis=1)  # (Q,)
        return float(max_per_query_token.sum())

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Retrieve the most relevant documents for *query*.

        Parameters
        ----------
        query : str
            Natural-language query.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[SearchResult]
            Ranked results with MaxSim scores.
        """
        if self.size == 0:
            return []

        query_emb = self._encode_query(query)

        scores = np.array(
            [self._maxsim_score(query_emb, doc_emb) for doc_emb in self._doc_embeddings],
            dtype=np.float32,
        )

        k = min(top_k, self.size)
        if k >= self.size:
            top_indices = np.argsort(-scores)[:k]
        else:
            top_indices = np.argpartition(-scores, k)[:k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: list[SearchResult] = []
        for idx in top_indices:
            results.append(
                SearchResult(
                    text=self._documents[idx],
                    score=float(scores[idx]),
                    metadata=self._metadata[idx],
                    vector_id=int(idx),
                )
            )
        return results
