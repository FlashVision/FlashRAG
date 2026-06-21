"""
SentenceTransformer embedding backend.

Wraps the ``sentence-transformers`` library for dense text embedding with
models like ``all-MiniLM-L6-v2``, ``bge-large-en-v1.5``, etc.
"""

from __future__ import annotations

import logging

import numpy as np

from flashrag.embeddings.base import BaseEmbedding
from flashrag.registry import EMBEDDINGS

logger = logging.getLogger(__name__)


@EMBEDDINGS.register("sentence_transformer")
class SentenceTransformerEmbedding(BaseEmbedding):
    """
    Dense text embeddings via ``sentence-transformers``.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. ``"all-MiniLM-L6-v2"``.
    device : str
        ``"cpu"`` or ``"cuda"``.
    cache_dir : str, optional
        Where to cache downloaded model files.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_dir: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.device = device
        self._model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)
        self._dimension: int = self._model.get_sentence_embedding_dimension()
        logger.info(
            f"Loaded SentenceTransformer '{model_name}' (dim={self._dimension}, device={device})"
        )

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
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)
