"""
Vision embedding backend using CLIP / SigLIP for image and text embeddings.

Enables multimodal RAG by embedding both images and text into a shared
vector space for cross-modal retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from PIL import Image as _PILImage

from flashrag.embeddings.base import BaseEmbedding
from flashrag.registry import EMBEDDINGS

logger = logging.getLogger(__name__)


@EMBEDDINGS.register("vision")
class VisionEmbedding(BaseEmbedding):
    """
    Image + text embeddings via CLIP or SigLIP.

    Parameters
    ----------
    model_name : str
        HuggingFace CLIP model name, e.g. ``"openai/clip-vit-base-patch32"``.
    device : str
        ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
    ) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError(
                "transformers with CLIP support is required. "
                "Install with: pip install transformers Pillow"
            )

        self.model_name = model_name
        self.device = device
        self._model = CLIPModel.from_pretrained(model_name).to(device)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model.eval()

        with torch.no_grad():
            dummy = self._processor(text=["test"], return_tensors="pt", padding=True)
            dummy = {k: v.to(device) for k, v in dummy.items() if k != "pixel_values"}
            out = self._model.get_text_features(**dummy)
            self._dimension = out.shape[-1]

        logger.info(
            f"Loaded CLIP model '{model_name}' (dim={self._dimension}, device={device})"
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode text strings using the CLIP text encoder."""
        all_vecs: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = self._processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items() if k != "pixel_values"}
            with torch.no_grad():
                vecs = self._model.get_text_features(**inputs)
            all_vecs.append(vecs.cpu().numpy())

        vectors = np.concatenate(all_vecs, axis=0).astype(np.float32)
        if normalize:
            vectors = self.l2_normalize(vectors)
        return vectors

    def encode_images(
        self,
        images: list[str | Path | _PILImage],
        batch_size: int = 16,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode images using the CLIP vision encoder."""
        try:
            from PIL import Image as PILImage
        except ImportError:
            raise ImportError("Pillow is required for image embedding: pip install Pillow")

        loaded: list = []
        for img in images:
            if isinstance(img, (str, Path)):
                loaded.append(PILImage.open(img).convert("RGB"))
            else:
                loaded.append(img)

        all_vecs: list[np.ndarray] = []
        for i in range(0, len(loaded), batch_size):
            batch = loaded[i: i + batch_size]
            inputs = self._processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items() if k != "input_ids"}
            with torch.no_grad():
                vecs = self._model.get_image_features(**inputs)
            all_vecs.append(vecs.cpu().numpy())

        vectors = np.concatenate(all_vecs, axis=0).astype(np.float32)
        if normalize:
            vectors = self.l2_normalize(vectors)
        return vectors
