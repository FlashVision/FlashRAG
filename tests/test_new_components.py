"""
Tests for new FlashRAG components: ColBERT, HyDE, query transforms,
semantic chunking, RAGAS evaluation, and GraphRAG.
"""

import numpy as np
import pytest


class TestColBERTRetriever:
    def test_index_and_search(self):
        from flashrag.retrieval.colbert import ColBERTRetriever

        retriever = ColBERTRetriever(dimension=32, max_doc_length=64, max_query_length=16)
        docs = [
            "The cat sat on the mat by the door",
            "Dogs are loyal and friendly companions",
            "Machine learning transforms raw data into insights",
        ]
        retriever.index(docs)
        assert retriever.size == 3

        results = retriever.search("cat mat", top_k=2)
        assert len(results) <= 2
        assert all(hasattr(r, "text") for r in results)
        assert all(hasattr(r, "score") for r in results)

    def test_empty_search(self):
        from flashrag.retrieval.colbert import ColBERTRetriever

        retriever = ColBERTRetriever(dimension=32)
        results = retriever.search("test query")
        assert results == []

    def test_metadata_preserved(self):
        from flashrag.retrieval.colbert import ColBERTRetriever

        retriever = ColBERTRetriever(dimension=32, max_doc_length=64)
        retriever.index(
            ["test document about search", "another different document here"],
            metadata=[{"source": "test.txt"}, {"source": "other.txt"}],
        )
        results = retriever.search("test document about search", top_k=2)
        assert len(results) >= 1
        sources = {r.metadata.get("source") for r in results}
        assert "test.txt" in sources or "other.txt" in sources


class _DummyEmbedding:
    """Minimal embedding model for offline testing."""

    dimension = 32

    def encode(self, texts, **kwargs):
        import hashlib
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dimension).astype(np.float32)
            result[i] = vec / max(np.linalg.norm(vec), 1e-12)
        return result


class _DummyRetriever:
    """Minimal in-memory retriever for offline testing (no FAISS needed)."""

    def __init__(self):
        self._docs = []
        self._vectors = None
        self._metadata = []

    @property
    def size(self):
        return len(self._docs)

    def add(self, vectors, documents, metadata=None):
        self._docs.extend(documents)
        self._metadata.extend(metadata or [{}] * len(documents))
        if self._vectors is None:
            self._vectors = vectors.copy()
        else:
            self._vectors = np.vstack([self._vectors, vectors])

    def search(self, query_vector, top_k=5):
        from flashrag.retrieval.vector_store import SearchResult
        if self.size == 0:
            return []
        q = query_vector.reshape(1, -1)
        norms_q = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / np.maximum(norms_q, 1e-12)
        norms_d = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        d = self._vectors / np.maximum(norms_d, 1e-12)
        scores = (q @ d.T).flatten()
        k = min(top_k, len(scores))
        indices = np.argsort(-scores)[:k]
        return [
            SearchResult(text=self._docs[i], score=float(scores[i]), metadata=self._metadata[i])
            for i in indices
        ]


class TestHyDERetriever:
    def test_index_and_search(self):
        from flashrag.retrieval.hyde import HyDERetriever

        base = _DummyRetriever()
        retriever = HyDERetriever(
            dimension=32,
            embedding_model=_DummyEmbedding(),
            base_retriever=base,
        )
        docs = [
            "Python is a programming language",
            "JavaScript runs in browsers",
            "Rust is a systems programming language",
        ]
        retriever.index(docs)
        assert retriever.size == 3

        results = retriever.search("programming language", top_k=2)
        assert len(results) <= 2

    def test_empty_search(self):
        from flashrag.retrieval.hyde import HyDERetriever

        base = _DummyRetriever()
        retriever = HyDERetriever(
            dimension=32,
            embedding_model=_DummyEmbedding(),
            base_retriever=base,
        )
        results = retriever.search("test")
        assert results == []


class TestQueryDecomposer:
    def test_decompose_compound_query(self):
        from flashrag.retrieval.query_transform import QueryDecomposer

        decomposer = QueryDecomposer()
        result = decomposer.decompose("What is Python and how does it compare to Java?")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_simple_query_unchanged(self):
        from flashrag.retrieval.query_transform import QueryDecomposer

        decomposer = QueryDecomposer()
        result = decomposer.decompose("What is Python?")
        assert isinstance(result, list)
        assert len(result) >= 1


class TestStepBackPrompter:
    def test_generate_stepback(self):
        from flashrag.retrieval.query_transform import StepBackPrompter

        prompter = StepBackPrompter()
        result = prompter.generate_stepback("What is the GDP of France in 2023?")
        assert isinstance(result, str)
        assert len(result) > 0


class TestMultiQueryGenerator:
    def test_generate_multiple(self):
        from flashrag.retrieval.query_transform import MultiQueryGenerator

        gen = MultiQueryGenerator(num_queries=3)
        result = gen.generate("How does machine learning work?")
        assert isinstance(result, list)
        assert len(result) >= 1


class TestSemanticChunker:
    def test_basic_chunking(self):
        from flashrag.data.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(min_chunk_size=20, max_chunk_size=200)
        text = (
            "Machine learning is a branch of artificial intelligence. "
            "It uses algorithms to learn from data. "
            "Deep learning is a subset of machine learning. "
            "Neural networks are the foundation of deep learning. "
            "Transformers have revolutionized natural language processing. "
            "They use self-attention mechanisms for better context understanding."
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert all(hasattr(c, "text") for c in chunks)
        full_text_from_chunks = " ".join(c.text for c in chunks)
        assert len(full_text_from_chunks) > 0

    def test_empty_text(self):
        from flashrag.data.semantic_chunker import SemanticChunker

        chunker = SemanticChunker()
        chunks = chunker.chunk("")
        assert chunks == []

    def test_metadata_passed(self):
        from flashrag.data.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(min_chunk_size=10, max_chunk_size=500)
        chunks = chunker.chunk("This is a test sentence.", metadata={"source": "test"})
        assert len(chunks) >= 1
        assert chunks[0].metadata.get("source") == "test"


class TestRAGASEvaluator:
    def test_context_precision(self):
        from flashrag.analytics.ragas import compute_context_precision

        questions = ["What is Python?"]
        contexts = [["Python is a programming language used for web development and data science"]]
        ground_truths = ["Python is a programming language"]
        score = compute_context_precision(questions, contexts, ground_truths)
        assert 0.0 <= score <= 1.0

    def test_context_recall(self):
        from flashrag.analytics.ragas import compute_context_recall

        questions = ["What is Python?"]
        contexts = [["Python is a programming language used widely"]]
        ground_truths = ["Python is a programming language"]
        score = compute_context_recall(questions, contexts, ground_truths)
        assert 0.0 <= score <= 1.0

    def test_answer_faithfulness(self):
        from flashrag.analytics.ragas import compute_answer_faithfulness

        answers = ["Python is great for data science"]
        contexts = [["Python is a versatile language great for data science and web dev"]]
        score = compute_answer_faithfulness(answers, contexts)
        assert 0.0 <= score <= 1.0

    def test_answer_relevance(self):
        from flashrag.analytics.ragas import compute_answer_relevance

        questions = ["What is machine learning?"]
        answers = ["Machine learning is a branch of AI that learns from data"]
        score = compute_answer_relevance(questions, answers)
        assert 0.0 <= score <= 1.0

    def test_evaluator_all_metrics(self):
        from flashrag.analytics.ragas import RAGASEvaluator

        evaluator = RAGASEvaluator()
        result = evaluator.evaluate(
            questions=["What is AI?"],
            answers=["AI is artificial intelligence"],
            contexts=[["Artificial intelligence is a field of computer science"]],
            ground_truths=["AI stands for artificial intelligence"],
        )
        assert "context_precision" in result
        assert "context_recall" in result
        assert "answer_faithfulness" in result
        assert "answer_relevance" in result


class TestGraphRAGPipeline:
    def test_entity_extraction(self):
        from flashrag.pipelines.graph_rag import GraphRAGPipeline

        pipeline = GraphRAGPipeline(chunk_size=200)
        entities = pipeline._extract_entities(
            "Albert Einstein developed the theory of relativity at Princeton University."
        )
        assert isinstance(entities, list)
        assert len(entities) >= 1

    def test_relation_extraction(self):
        from flashrag.pipelines.graph_rag import GraphRAGPipeline

        pipeline = GraphRAGPipeline(chunk_size=200)
        text = "Albert Einstein developed the theory of relativity."
        entities = pipeline._extract_entities(text)
        relations = pipeline._extract_relations(text, entities)
        assert isinstance(relations, list)

    def test_graph_stats_empty(self):
        from flashrag.pipelines.graph_rag import GraphRAGPipeline

        pipeline = GraphRAGPipeline(chunk_size=200)
        stats = pipeline.get_graph_stats()
        assert "num_entities" in stats
        assert "num_chunks" in stats
        assert stats["num_chunks"] == 0


class TestRegistryNewComponents:
    def test_new_retrievers_registered(self):
        from flashrag.registry import RETRIEVERS, auto_register

        auto_register()
        assert "colbert" in RETRIEVERS
        assert "hyde" in RETRIEVERS

    def test_new_pipeline_registered(self):
        from flashrag.registry import PIPELINES, auto_register

        auto_register()
        assert "graph_rag" in PIPELINES
