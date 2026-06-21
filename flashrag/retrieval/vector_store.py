"""
FAISS-backed dense vector store.

Supports add / search / save / load with cosine or L2 distance.
Stores document metadata alongside vectors for end-to-end retrieval.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from flashrag.registry import RETRIEVERS

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector_id: int = -1


@RETRIEVERS.register("faiss")
class VectorStore:
    """
    In-memory vector store backed by FAISS.

    Parameters
    ----------
    dimension : int
        Dimensionality of stored vectors.
    metric : str
        ``"cosine"`` (inner product on normalized vectors) or ``"l2"``.
    use_gpu : bool
        Attempt to use FAISS GPU resources if available.
    """

    def __init__(
        self,
        dimension: int = 384,
        metric: str = "cosine",
        use_gpu: bool = False,
    ) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")

        self.dimension = dimension
        self.metric = metric
        self._faiss = faiss

        if metric == "cosine":
            self._index = faiss.IndexFlatIP(dimension)
        elif metric == "l2":
            self._index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported metric '{metric}'. Use 'cosine' or 'l2'.")

        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
                logger.info("FAISS index moved to GPU")
            except Exception:
                logger.warning("GPU FAISS unavailable, falling back to CPU")

        self._documents: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        logger.info(f"VectorStore initialized (dim={dimension}, metric={metric})")

    @property
    def size(self) -> int:
        return self._index.ntotal

    def add(
        self,
        vectors: np.ndarray,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add vectors and their associated documents to the store."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[0] != len(documents):
            raise ValueError(
                f"Vector count ({vectors.shape[0]}) != document count ({len(documents)})"
            )
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension ({vectors.shape[1]}) != store dimension ({self.dimension})"
            )

        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            vectors = vectors / norms

        self._index.add(vectors)
        self._documents.extend(documents)
        self._metadata.extend(metadata or [{}] * len(documents))
        logger.debug(f"Added {len(documents)} vectors (total: {self.size})")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Search for the ``top_k`` most similar vectors."""
        if self.size == 0:
            return []

        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if self.metric == "cosine":
            norms = np.linalg.norm(query_vector, axis=1, keepdims=True)
            query_vector = query_vector / np.maximum(norms, 1e-12)

        k = min(top_k, self.size)
        distances, indices = self._index.search(query_vector, k)

        results: List[SearchResult] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                SearchResult(
                    text=self._documents[idx],
                    score=float(score),
                    metadata=self._metadata[idx],
                    vector_id=int(idx),
                )
            )
        return results

    def batch_search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 5,
    ) -> List[List[SearchResult]]:
        """Search for multiple queries at once."""
        query_vectors = np.asarray(query_vectors, dtype=np.float32)
        if self.metric == "cosine":
            norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
            query_vectors = query_vectors / np.maximum(norms, 1e-12)

        k = min(top_k, self.size)
        distances, indices = self._index.search(query_vectors, k)

        all_results: List[List[SearchResult]] = []
        for dists, idxs in zip(distances, indices):
            results = []
            for score, idx in zip(dists, idxs):
                if idx < 0:
                    continue
                results.append(
                    SearchResult(
                        text=self._documents[idx],
                        score=float(score),
                        metadata=self._metadata[idx],
                        vector_id=int(idx),
                    )
                )
            all_results.append(results)
        return all_results

    def remove(self, vector_ids: List[int]) -> None:
        """Remove vectors by their IDs (rebuilds the index)."""
        keep_mask = np.ones(self.size, dtype=bool)
        for vid in vector_ids:
            if 0 <= vid < self.size:
                keep_mask[vid] = False

        all_vectors = self._faiss.rev_swig_ptr(
            self._index.get_xb(), self.size * self.dimension
        ).reshape(self.size, self.dimension).copy()

        kept_vectors = all_vectors[keep_mask]
        kept_docs = [d for d, m in zip(self._documents, keep_mask) if m]
        kept_meta = [d for d, m in zip(self._metadata, keep_mask) if m]

        self._index.reset()
        self._documents = []
        self._metadata = []

        if len(kept_vectors) > 0:
            self._index.add(kept_vectors)
            self._documents = kept_docs
            self._metadata = kept_meta

    def save(self, path: str | Path) -> None:
        """Persist the index and metadata to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        index_path = path / "index.faiss"
        meta_path = path / "metadata.json"

        cpu_index = self._index
        try:
            cpu_index = self._faiss.index_gpu_to_cpu(self._index)
        except Exception:
            pass

        self._faiss.write_index(cpu_index, str(index_path))
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "documents": self._documents,
                    "metadata": self._metadata,
                    "dimension": self.dimension,
                    "metric": self.metric,
                },
                f,
            )
        logger.info(f"VectorStore saved to {path} ({self.size} vectors)")

    @classmethod
    def load(cls, path: str | Path, use_gpu: bool = False) -> "VectorStore":
        """Load a previously saved vector store."""
        import faiss

        path = Path(path)
        index_path = path / "index.faiss"
        meta_path = path / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        with open(meta_path) as f:
            data = json.load(f)

        store = cls(
            dimension=data["dimension"],
            metric=data.get("metric", "cosine"),
            use_gpu=use_gpu,
        )
        store._index = faiss.read_index(str(index_path))
        store._documents = data["documents"]
        store._metadata = data["metadata"]

        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                store._index = faiss.index_cpu_to_gpu(res, 0, store._index)
            except Exception:
                pass

        logger.info(f"VectorStore loaded from {path} ({store.size} vectors)")
        return store
