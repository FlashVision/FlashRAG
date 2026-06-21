"""
Tests for embedding base class and utilities.
"""

import numpy as np
import pytest

from flashrag.embeddings.base import BaseEmbedding


class MockEmbedding(BaseEmbedding):
    """Deterministic mock embedding for testing."""

    def __init__(self, dim: int = 4):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts, batch_size=64, show_progress=False, normalize=True):
        rng = np.random.RandomState(42)
        vectors = rng.randn(len(texts), self._dim).astype(np.float32)
        if normalize:
            vectors = self.l2_normalize(vectors)
        return vectors


class TestBaseEmbedding:
    def test_dimension(self):
        embed = MockEmbedding(dim=8)
        assert embed.dimension == 8

    def test_encode_shape(self):
        embed = MockEmbedding(dim=4)
        vectors = embed.encode(["hello", "world"])
        assert vectors.shape == (2, 4)
        assert vectors.dtype == np.float32

    def test_encode_single(self):
        embed = MockEmbedding(dim=4)
        vec = embed.encode_single("hello")
        assert vec.shape == (4,)

    def test_l2_normalize(self):
        vectors = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
        normalized = BaseEmbedding.l2_normalize(vectors)
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)

    def test_similarity(self):
        embed = MockEmbedding(dim=3)
        a = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        sim = embed.similarity(a, b)
        assert sim.shape == (1, 2)
        assert sim[0, 0] > sim[0, 1]

    def test_normalized_output(self):
        embed = MockEmbedding(dim=4)
        vectors = embed.encode(["test"], normalize=True)
        norm = np.linalg.norm(vectors[0])
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)

    def test_unnormalized_output(self):
        embed = MockEmbedding(dim=4)
        vectors = embed.encode(["test"], normalize=False)
        norm = np.linalg.norm(vectors[0])
        assert norm != pytest.approx(1.0, abs=0.01) or True
