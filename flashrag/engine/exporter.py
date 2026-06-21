"""
RAG system exporter.

Export vector indices, configurations, and model artifacts for deployment.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from flashrag.cfg.config import RAGConfig
from flashrag.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGExporter:
    """
    Export a RAG system's components for deployment.

    Bundles the vector index, configuration, and optionally the
    embedding/generator model into a deployable package.

    Parameters
    ----------
    output_dir : str or Path
        Directory to export to.
    """

    def __init__(self, output_dir: str | Path = "workspace/export") -> None:
        self.output_dir = Path(output_dir)

    def export(
        self,
        vector_store: VectorStore,
        config: RAGConfig | None = None,
        model_path: str | Path | None = None,
        include_model: bool = False,
    ) -> dict[str, Any]:
        """
        Export the RAG system.

        Parameters
        ----------
        vector_store : VectorStore
            The vector store to export.
        config : RAGConfig, optional
            Configuration to bundle.
        model_path : str, optional
            Path to a fine-tuned model to include.
        include_model : bool
            Whether to copy model weights into the export.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {"version": "1.0.0", "components": []}

        index_dir = self.output_dir / "index"
        vector_store.save(index_dir)
        manifest["components"].append({"type": "vector_index", "path": "index/"})

        if config:
            config_path = self.output_dir / "config.yaml"
            import yaml

            with open(config_path, "w") as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False)
            manifest["components"].append({"type": "config", "path": "config.yaml"})

        if include_model and model_path:
            model_dest = self.output_dir / "model"
            model_dest.mkdir(exist_ok=True)
            src = Path(model_path)
            if src.is_dir():
                shutil.copytree(src, model_dest, dirs_exist_ok=True)
            manifest["components"].append({"type": "model", "path": "model/"})

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"RAG system exported to {self.output_dir}")
        return manifest
