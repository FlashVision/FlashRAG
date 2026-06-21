"""
Abstract base class for all embedding backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedding(ABC):
    """
    Interface every embedding backend must implement.

    Subclasses provide ``encode`` which converts texts (or images) into
    dense vectors returned as a NumPy array of shape ``(N, dim)``.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the output vectors."""

    @abstractmethod
    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into an ``(N, dim)`` float32 array.

        Parameters
        ----------
        texts : list[str]
            Input strings to embed.
        batch_size : int
            Mini-batch size for encoding.
        show_progress : bool
            Show a tqdm progress bar.
        normalize : bool
            L2-normalize each vector (required for cosine similarity via dot product).
        """

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Convenience: encode one string → 1-D vector."""
        return self.encode([text], normalize=normalize)[0]

    @staticmethod
    def l2_normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return vectors / norms

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity matrix between two sets of vectors."""
        a_norm = self.l2_normalize(a)
        b_norm = self.l2_normalize(b)
        return a_norm @ b_norm.T
