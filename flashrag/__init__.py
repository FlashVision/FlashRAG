"""
FlashRAG — Production-grade Retrieval-Augmented Generation.

Vector databases, document retrieval, embedding models, knowledge-grounded QA.
"""

__version__ = "1.0.0"

from flashrag.analytics.benchmark import Benchmark
from flashrag.cfg.config import RAGConfig, get_config
from flashrag.pipelines.basic_rag import BasicRAGPipeline
from flashrag.registry import EMBEDDINGS, GENERATORS, PIPELINES, RERANKERS, RETRIEVERS
from flashrag.retrieval.hybrid import HybridSearch
from flashrag.retrieval.vector_store import VectorStore
from flashrag.solutions.document_qa import DocumentQA
from flashrag.solutions.knowledge_base import KnowledgeBase


class FlashRAG:
    """
    Facade entry point for FlashRAG.

    Provides convenience factory methods for creating RAG systems.
    """

    version = __version__

    @staticmethod
    def create_pipeline(
        pipeline: str = "basic_rag",
        embedding_model: str = "all-MiniLM-L6-v2",
        generator_model: str = "gpt2",
        **kwargs,
    ):
        """Create a RAG pipeline by name."""
        from flashrag.registry import PIPELINES, auto_register

        auto_register()
        pipeline_cls = PIPELINES.get(pipeline)
        return pipeline_cls(
            embedding_model=embedding_model,
            generator_model=generator_model,
            **kwargs,
        )

    @staticmethod
    def document_qa(**kwargs) -> "DocumentQA":
        """Create a DocumentQA instance."""
        return DocumentQA(**kwargs)

    @staticmethod
    def knowledge_base(**kwargs) -> "KnowledgeBase":
        """Create a KnowledgeBase instance."""
        return KnowledgeBase(**kwargs)


__all__ = [
    "__version__",
    "FlashRAG",
    "VectorStore",
    "HybridSearch",
    "BasicRAGPipeline",
    "DocumentQA",
    "KnowledgeBase",
    "Benchmark",
    "RAGConfig",
    "get_config",
    "EMBEDDINGS",
    "RETRIEVERS",
    "GENERATORS",
    "PIPELINES",
    "RERANKERS",
]
