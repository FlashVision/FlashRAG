"""
OpenAI API embedding backend.

Uses the OpenAI ``text-embedding-3-small`` or ``text-embedding-3-large``
models via the official Python client.  Requires ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from flashrag.embeddings.base import BaseEmbedding
from flashrag.registry import EMBEDDINGS

logger = logging.getLogger(__name__)

_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


@EMBEDDINGS.register("openai")
class OpenAIEmbedding(BaseEmbedding):
    """
    Dense embeddings via the OpenAI Embeddings API.

    Parameters
    ----------
    model_name : str
        OpenAI embedding model, e.g. ``"text-embedding-3-small"``.
    api_key : str, optional
        API key (falls back to ``OPENAI_API_KEY`` env var).
    max_retries : int
        Number of retries on transient errors.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        max_retries: int = 3,
    ) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai is required for OpenAI embeddings. "
                "Install with: pip install 'flashrag[openai]'"
            )

        self.model_name = model_name
        self._dimension = _MODEL_DIMENSIONS.get(model_name, 1536)
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key=")
        self._client = openai.OpenAI(api_key=key, max_retries=max_retries)
        logger.info(f"Initialized OpenAI embedding model '{model_name}' (dim={self._dimension})")

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch = [t.replace("\n", " ") for t in batch]
            response = self._client.embeddings.create(input=batch, model=self.model_name)
            batch_vecs = [item.embedding for item in response.data]
            all_embeddings.extend(batch_vecs)

        vectors = np.array(all_embeddings, dtype=np.float32)
        if normalize:
            vectors = self.l2_normalize(vectors)
        return vectors
