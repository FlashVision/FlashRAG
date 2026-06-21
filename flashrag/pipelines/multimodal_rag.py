"""
Multimodal RAG pipeline for text + image retrieval.

Combines text and image embeddings in a shared CLIP vector space,
enabling cross-modal retrieval (text query → image results and vice versa).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from flashrag.generation.generator import RAGGenerator
from flashrag.pipelines.basic_rag import RAGResult
from flashrag.registry import PIPELINES
from flashrag.retrieval.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


@PIPELINES.register("multimodal_rag")
class MultimodalRAGPipeline:
    """
    Multimodal RAG pipeline supporting text and image retrieval.

    Uses CLIP/SigLIP to embed both text and images into a shared space,
    then retrieves relevant content regardless of modality.

    Parameters
    ----------
    vision_model : str
        CLIP model name for shared text-image embeddings.
    generator_model : str
        LLM for answer generation.
    top_k : int
        Number of results to retrieve.
    device : str
        Device for models.
    """

    def __init__(
        self,
        vision_model: str = "openai/clip-vit-base-patch32",
        generator_model: str = "gpt2",
        top_k: int = 5,
        device: str = "cpu",
    ) -> None:
        from flashrag.embeddings.vision_embeddings import VisionEmbedding

        self._embedder = VisionEmbedding(model_name=vision_model, device=device)
        self._vector_store = VectorStore(dimension=self._embedder.dimension, metric="cosine")
        self._generator = RAGGenerator(model_name=generator_model, device=device)
        self.top_k = top_k

        self._documents: List[str] = []
        self._modalities: List[str] = []
        logger.info("MultimodalRAGPipeline initialized")

    def index_texts(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Index text documents using CLIP text encoder."""
        meta = metadata or [{"modality": "text"}] * len(texts)
        for m in meta:
            m.setdefault("modality", "text")

        vectors = self._embedder.encode(texts)
        self._vector_store.add(vectors, texts, meta)
        self._documents.extend(texts)
        self._modalities.extend(["text"] * len(texts))
        return len(texts)

    def index_images(
        self,
        image_paths: List[str | Path],
        captions: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Index images using CLIP vision encoder."""
        vectors = self._embedder.encode_images(image_paths)

        display_texts = captions or [f"[Image: {Path(p).name}]" for p in image_paths]
        meta = metadata or [{}] * len(image_paths)
        for i, m in enumerate(meta):
            m["modality"] = "image"
            m["image_path"] = str(image_paths[i])
            if captions and i < len(captions):
                m["caption"] = captions[i]

        self._vector_store.add(vectors, display_texts, meta)
        self._documents.extend(display_texts)
        self._modalities.extend(["image"] * len(image_paths))
        return len(image_paths)

    def search_by_text(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Search the multimodal index with a text query."""
        k = top_k or self.top_k
        query_vec = self._embedder.encode([query])[0]
        return self._vector_store.search(query_vec, top_k=k)

    def search_by_image(
        self, image_path: str | Path, top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """Search the multimodal index with an image query."""
        k = top_k or self.top_k
        query_vec = self._embedder.encode_images([image_path])[0]
        return self._vector_store.search(query_vec, top_k=k)

    def run(
        self,
        question: str,
        top_k: Optional[int] = None,
        **gen_kwargs: Any,
    ) -> RAGResult:
        """Run text-query multimodal retrieval + generation."""
        results = self.search_by_text(question, top_k=top_k)

        text_contexts = []
        for r in results:
            modality = r.metadata.get("modality", "text")
            if modality == "image":
                caption = r.metadata.get("caption", r.text)
                text_contexts.append(f"[Image] {caption}")
            else:
                text_contexts.append(r.text)

        gen_result = self._generator.generate(
            question=question,
            contexts=text_contexts,
            **gen_kwargs,
        )

        return RAGResult(
            answer=gen_result.answer,
            contexts=text_contexts,
            sources=[r.metadata for r in results],
            scores=[r.score for r in results],
            metadata={"pipeline": "multimodal"},
        )
