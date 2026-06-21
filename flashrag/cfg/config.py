"""
Hierarchical configuration for FlashRAG.

Supports loading from YAML files, CLI overrides, and programmatic construction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    backend: str = "sentence_transformer"
    dimension: int = 384
    batch_size: int = 64
    device: str = "cpu"
    normalize: bool = True
    cache_dir: str | None = None


@dataclass
class RetrieverConfig:
    backend: str = "faiss"
    top_k: int = 5
    index_path: str | None = None
    use_bm25: bool = False
    use_hybrid: bool = False
    hybrid_alpha: float = 0.5
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 3


@dataclass
class GeneratorConfig:
    model_name: str = "gpt2"
    backend: str = "huggingface"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    device: str = "cpu"
    prompt_template: str = "default"


@dataclass
class DataConfig:
    docs_path: str | None = None
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunk_strategy: str = "recursive"
    supported_formats: list[str] = field(
        default_factory=lambda: ["pdf", "html", "md", "csv", "txt"]
    )


@dataclass
class AnalyticsConfig:
    metrics: list[str] = field(default_factory=lambda: ["recall@5", "mrr", "ndcg@10"])
    output_dir: str = "workspace/eval"


@dataclass
class RAGConfig:
    project_name: str = "flashrag_experiment"
    seed: int = 42
    output_dir: str = "workspace"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    data: DataConfig = field(default_factory=DataConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RAGConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> RAGConfig:
        cfg = cls()
        simple_fields = {"project_name", "seed", "output_dir"}
        for k in simple_fields:
            if k in d:
                setattr(cfg, k, d[k])

        section_map = {
            "embedding": (cfg.embedding, EmbeddingConfig),
            "retriever": (cfg.retriever, RetrieverConfig),
            "generator": (cfg.generator, GeneratorConfig),
            "data": (cfg.data, DataConfig),
            "analytics": (cfg.analytics, AnalyticsConfig),
        }
        for section_name, (obj, _dc) in section_map.items():
            if section_name in d and isinstance(d[section_name], dict):
                for k, v in d[section_name].items():
                    if hasattr(obj, k):
                        setattr(obj, k, v)
        return cfg

    def merge_cli(self, overrides: dict[str, Any]) -> None:
        """Apply dot-separated CLI overrides like ``retriever.top_k=10``."""
        for key, value in overrides.items():
            parts = key.split(".")
            obj: Any = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

    def to_dict(self) -> dict[str, Any]:
        import dataclasses

        def _asdict(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj):
                return {k: _asdict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        return _asdict(self)


def get_config(path: str | Path | None = None) -> RAGConfig:
    if path is not None:
        return RAGConfig.from_yaml(path)
    env_path = os.environ.get("FLASHRAG_CONFIG")
    if env_path:
        return RAGConfig.from_yaml(env_path)
    return RAGConfig()
