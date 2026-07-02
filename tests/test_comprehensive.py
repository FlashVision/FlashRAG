"""Comprehensive tests for FlashRAG: retrievers, chunkers, generation, pipelines, evaluation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Retrieval: HybridSearch
# ---------------------------------------------------------------------------


class TestHybridSearch:
    def _make_hybrid(self):
        from flashrag.retrieval.hybrid import HybridSearch

        class FakeEmbed:
            dimension = 4

            def encode(self, texts, show_progress=False):
                rng = np.random.RandomState(0)
                return rng.randn(len(texts), self.dimension).astype(np.float32)

        return HybridSearch(embedding_model=FakeEmbed(), alpha=0.5, rrf_k=60)

    def test_index_and_search(self):
        hs = self._make_hybrid()
        docs = ["cat sat on the mat", "dogs are loyal", "machine learning is powerful"]
        hs.index(docs)
        assert hs.size == 3
        results = hs.search("cat", top_k=2, dense_top_k=2, sparse_top_k=2)
        assert len(results) == 2
        assert all(hasattr(r, "text") for r in results)

    def test_empty_search(self):
        hs = self._make_hybrid()
        results = hs.search("hello", top_k=5)
        assert results == []

    def test_add_documents(self):
        hs = self._make_hybrid()
        hs.index(["first doc"])
        assert hs.size == 1
        hs.add_documents(["second doc", "third doc"])
        assert hs.size == 3

    def test_alpha_weight(self):
        from flashrag.retrieval.hybrid import HybridSearch

        class FakeEmbed:
            dimension = 4

            def encode(self, texts, show_progress=False):
                rng = np.random.RandomState(7)
                return rng.randn(len(texts), 4).astype(np.float32)

        hs_dense = HybridSearch(embedding_model=FakeEmbed(), alpha=1.0)
        hs_sparse = HybridSearch(embedding_model=FakeEmbed(), alpha=0.0)
        docs = [
            "alpha beta gamma",
            "delta epsilon zeta",
            "alpha theta iota",
            "kappa lambda mu",
            "nu xi omicron",
        ]
        hs_dense.index(docs)
        hs_sparse.index(docs)
        r1 = hs_dense.search("alpha", top_k=2, dense_top_k=4, sparse_top_k=4)
        r2 = hs_sparse.search("alpha", top_k=2, dense_top_k=4, sparse_top_k=4)
        assert len(r1) > 0
        assert len(r2) > 0


class TestReciprocalRankFusion:
    def test_basic_fusion(self):
        from flashrag.retrieval.hybrid import reciprocal_rank_fusion
        from flashrag.retrieval.vector_store import SearchResult

        list1 = [
            SearchResult(text="doc_a", score=0.9, metadata={}),
            SearchResult(text="doc_b", score=0.8, metadata={}),
        ]
        list2 = [
            SearchResult(text="doc_b", score=0.95, metadata={}),
            SearchResult(text="doc_c", score=0.7, metadata={}),
        ]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        assert len(fused) == 3
        assert fused[0].score > fused[-1].score

    def test_custom_weights(self):
        from flashrag.retrieval.hybrid import reciprocal_rank_fusion
        from flashrag.retrieval.vector_store import SearchResult

        list1 = [SearchResult(text="only_in_1", score=1.0, metadata={})]
        list2 = [SearchResult(text="only_in_2", score=1.0, metadata={})]
        fused = reciprocal_rank_fusion([list1, list2], k=60, weights=[10.0, 1.0])
        assert fused[0].text == "only_in_1"


# ---------------------------------------------------------------------------
# Retrieval: ColBERT
# ---------------------------------------------------------------------------


class TestColBERTRetriever:
    def test_index_and_search(self):
        from flashrag.retrieval.colbert import ColBERTRetriever

        retriever = ColBERTRetriever(dimension=32, device="cpu")
        docs = ["the quick brown fox", "lazy dog sleeps", "python programming"]
        retriever.index(docs)
        assert retriever.size == 3

        results = retriever.search("fox", top_k=2)
        assert len(results) == 2
        assert results[0].score >= results[1].score

    def test_empty_search(self):
        from flashrag.retrieval.colbert import ColBERTRetriever

        retriever = ColBERTRetriever(dimension=16)
        results = retriever.search("anything")
        assert results == []

    def test_maxsim_score(self):
        from flashrag.retrieval.colbert import ColBERTRetriever

        q = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        d = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        score = ColBERTRetriever._maxsim_score(q, d)
        assert score == pytest.approx(2.0, abs=1e-5)

    def test_metadata_preserved(self):
        from flashrag.retrieval.colbert import ColBERTRetriever

        retriever = ColBERTRetriever(dimension=16)
        retriever.index(["test doc"], metadata=[{"source": "test.txt"}])
        results = retriever.search("test", top_k=1)
        assert results[0].metadata["source"] == "test.txt"


# ---------------------------------------------------------------------------
# Retrieval: HyDE
# ---------------------------------------------------------------------------


class TestHyDERetriever:
    def test_template_fallback(self):
        from flashrag.retrieval.hyde import HyDERetriever

        class FakeEmbed:
            dimension = 4

            def encode(self, texts, show_progress=False):
                return np.random.RandomState(42).randn(len(texts), 4).astype(np.float32)

        retriever = HyDERetriever(embedding_model=FakeEmbed(), dimension=4)
        docs = ["quantum physics explains particles", "chemistry of molecules"]
        retriever.index(docs)
        assert retriever.size == 2

        results = retriever.search("what is quantum mechanics?", top_k=1)
        assert len(results) == 1

    def test_empty_store(self):
        from flashrag.retrieval.hyde import HyDERetriever

        retriever = HyDERetriever(dimension=4)
        results = retriever.search("anything")
        assert results == []

    def test_generate_with_template(self):
        from flashrag.retrieval.hyde import HyDERetriever

        retriever = HyDERetriever(num_hypothetical=2, dimension=4)
        hypotheticals = retriever._generate_with_template("What is Python?")
        assert len(hypotheticals) == 2
        assert "Python" in hypotheticals[0]


# ---------------------------------------------------------------------------
# Retrieval: Reranker (mocked transformers)
# ---------------------------------------------------------------------------


class TestCrossEncoderReranker:
    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForSequenceClassification")
    def test_rerank(self, mock_model_cls, mock_tok_cls):
        import torch

        from flashrag.retrieval.vector_store import SearchResult

        mock_tok = MagicMock()
        tok_output = MagicMock()
        tok_output.to.return_value = {
            "input_ids": torch.zeros(2, 10, dtype=torch.long),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
        }
        mock_tok.return_value = tok_output
        mock_tok_cls.from_pretrained.return_value = mock_tok

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        logits = torch.tensor([[0.9], [0.1]])
        mock_model.return_value = MagicMock(logits=logits)
        mock_model_cls.from_pretrained.return_value = mock_model

        from flashrag.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name="test", device="cpu")
        results = [
            SearchResult(text="low relevance", score=0.5, metadata={}),
            SearchResult(text="high relevance", score=0.3, metadata={}),
        ]
        reranked = reranker.rerank("test query", results)
        assert len(reranked) == 2
        assert reranked[0].score >= reranked[1].score

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForSequenceClassification")
    def test_rerank_empty(self, mock_model_cls, mock_tok_cls):
        mock_tok_cls.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model

        from flashrag.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name="test", device="cpu")
        assert reranker.rerank("query", []) == []


# ---------------------------------------------------------------------------
# Query Transformation
# ---------------------------------------------------------------------------


class TestQueryDecomposer:
    def test_simple_query(self):
        from flashrag.retrieval.query_transform import QueryDecomposer

        qd = QueryDecomposer()
        result = qd.decompose("What is machine learning?")
        assert len(result) >= 1
        assert "machine learning" in result[0].lower()

    def test_compound_query(self):
        from flashrag.retrieval.query_transform import QueryDecomposer

        qd = QueryDecomposer()
        result = qd.decompose("What is Python and how does it compare to Java?")
        assert len(result) >= 2

    def test_invalid_method(self):
        from flashrag.retrieval.query_transform import QueryDecomposer

        with pytest.raises(ValueError, match="Unknown"):
            QueryDecomposer(method="invalid")


class TestStepBackPrompter:
    def test_generalize_specific_date(self):
        from flashrag.retrieval.query_transform import StepBackPrompter

        sb = StepBackPrompter()
        result = sb.generate_stepback("What happened in 2023?")
        assert "2023" not in result

    def test_broadens_narrow_query(self):
        from flashrag.retrieval.query_transform import StepBackPrompter

        sb = StepBackPrompter()
        result = sb.generate_stepback("How do neural networks work?")
        assert len(result) > 0
        assert result != "How do neural networks work?"


class TestMultiQueryGenerator:
    def test_generates_variants(self):
        from flashrag.retrieval.query_transform import MultiQueryGenerator

        mqg = MultiQueryGenerator(num_queries=3)
        variants = mqg.generate("What is the best way to learn Python?")
        assert len(variants) >= 2
        assert variants[0] == "What is the best way to learn Python?"

    def test_original_always_first(self):
        from flashrag.retrieval.query_transform import MultiQueryGenerator

        mqg = MultiQueryGenerator(num_queries=5)
        q = "explain deep learning"
        variants = mqg.generate(q)
        assert variants[0] == q


class TestQueryRouter:
    def test_route_factual(self):
        from flashrag.retrieval.query_transform import QueryRouter

        mock_r = MagicMock()
        mock_r.search.return_value = []
        router = QueryRouter(retrievers={"bm25": mock_r, "dense": mock_r})
        router.set_routing({"factual": "bm25", "procedural": "dense"})
        name = router.route("What is the capital of France?")
        assert name == "bm25"

    def test_route_procedural(self):
        from flashrag.retrieval.query_transform import QueryRouter

        mock_r = MagicMock()
        router = QueryRouter(retrievers={"bm25": mock_r, "dense": mock_r})
        router.set_routing({"factual": "bm25", "procedural": "dense"})
        name = router.route("How do I install Python?")
        assert name == "dense"

    def test_empty_retrievers_raises(self):
        from flashrag.retrieval.query_transform import QueryRouter

        with pytest.raises(ValueError, match="At least one"):
            QueryRouter(retrievers={})


# ---------------------------------------------------------------------------
# Data: Chunkers
# ---------------------------------------------------------------------------


class TestRecursiveChunker:
    def test_short_text_single_chunk(self):
        from flashrag.data.chunkers import RecursiveChunker

        chunker = RecursiveChunker(chunk_size=1000)
        chunks = chunker.chunk("Short text.")
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_long_text_multiple_chunks(self):
        from flashrag.data.chunkers import RecursiveChunker

        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        text = "Word " * 100
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        for c in chunks:
            assert c.chunk_index >= 0

    def test_metadata_propagated(self):
        from flashrag.data.chunkers import RecursiveChunker

        chunker = RecursiveChunker(chunk_size=50)
        chunks = chunker.chunk("A " * 100, metadata={"source": "test.txt"})
        assert all(c.metadata.get("source") == "test.txt" for c in chunks)


class TestSentenceChunker:
    def test_sentence_splitting(self):
        from flashrag.data.chunkers import SentenceChunker

        chunker = SentenceChunker(chunk_size=50, sentence_overlap=1)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_single_sentence(self):
        from flashrag.data.chunkers import SentenceChunker

        chunker = SentenceChunker(chunk_size=1000)
        chunks = chunker.chunk("Only one sentence here.")
        assert len(chunks) == 1


class TestFixedChunker:
    def test_fixed_size(self):
        from flashrag.data.chunkers import FixedChunker

        chunker = FixedChunker(chunk_size=10, chunk_overlap=0)
        chunks = chunker.chunk("0123456789ABCDEF")
        assert len(chunks) == 2
        assert chunks[0].text == "0123456789"

    def test_overlap(self):
        from flashrag.data.chunkers import FixedChunker

        chunker = FixedChunker(chunk_size=10, chunk_overlap=5)
        chunks = chunker.chunk("0123456789ABCDEFGHIJ")
        assert len(chunks) >= 3


# ---------------------------------------------------------------------------
# Semantic Chunker
# ---------------------------------------------------------------------------


class TestSemanticChunker:
    def test_chunks_text(self):
        from flashrag.data.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(min_chunk_size=10, max_chunk_size=200)
        text = (
            "Machine learning is a subset of AI. "
            "It uses data to train models. "
            "Deep learning uses neural networks. "
            "Cooking requires fresh ingredients. "
            "A recipe specifies the steps."
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert all(c.chunk_index >= 0 for c in chunks)

    def test_single_sentence(self):
        from flashrag.data.semantic_chunker import SemanticChunker

        chunker = SemanticChunker()
        chunks = chunker.chunk("One sentence only.")
        assert len(chunks) == 1

    def test_empty_text(self):
        from flashrag.data.semantic_chunker import SemanticChunker

        chunker = SemanticChunker()
        chunks = chunker.chunk("")
        assert chunks == []

    def test_with_embedding_model(self):
        from flashrag.data.semantic_chunker import SemanticChunker

        class FakeEmbed:
            def encode(self, texts):
                return np.random.RandomState(0).randn(len(texts), 8).astype(np.float32)

        chunker = SemanticChunker(embedding_model=FakeEmbed(), min_chunk_size=10)
        text = "First topic. Second topic. Third topic. Fourth topic."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Document Loaders
# ---------------------------------------------------------------------------


class TestDocumentLoaders:
    def test_plain_text(self, tmp_path):
        from flashrag.data.loaders import PlainTextLoader

        p = tmp_path / "test.txt"
        p.write_text("Hello world")
        docs = PlainTextLoader().load(p)
        assert len(docs) == 1
        assert docs[0].text == "Hello world"
        assert docs[0].metadata["type"] == "text"

    def test_html_loader(self, tmp_path):
        from flashrag.data.loaders import HTMLLoader

        p = tmp_path / "test.html"
        p.write_text("<html><body><p>Hello &amp; world</p></body></html>")
        docs = HTMLLoader().load(p)
        assert len(docs) == 1
        assert "Hello & world" in docs[0].text
        assert "<p>" not in docs[0].text

    def test_markdown_loader(self, tmp_path):
        from flashrag.data.loaders import MarkdownLoader

        p = tmp_path / "test.md"
        p.write_text("# Title\n\nContent here.\n\n## Section\n\nMore content.")
        docs = MarkdownLoader().load(p, split_on_headers=True)
        assert len(docs) == 2
        assert docs[0].metadata["header"] == "Title"

    def test_csv_loader(self, tmp_path):
        from flashrag.data.loaders import CSVLoader

        p = tmp_path / "data.csv"
        p.write_text("name,value\nalpha,1\nbeta,2\n")
        docs = CSVLoader().load(p)
        assert len(docs) == 2
        assert "alpha" in docs[0].text

    def test_document_loader_dispatch(self, tmp_path):
        from flashrag.data.loaders import DocumentLoader

        p = tmp_path / "test.txt"
        p.write_text("dispatch test")
        loader = DocumentLoader()
        docs = loader.load(p)
        assert len(docs) == 1
        assert docs[0].text == "dispatch test"

    def test_directory_loading(self, tmp_path):
        from flashrag.data.loaders import DocumentLoader

        (tmp_path / "a.txt").write_text("doc a")
        (tmp_path / "b.txt").write_text("doc b")
        loader = DocumentLoader()
        docs = loader.load(tmp_path)
        assert len(docs) == 2


# ---------------------------------------------------------------------------
# Embeddings (mock SentenceTransformer, OpenAI, CLIP)
# ---------------------------------------------------------------------------


class TestSentenceTransformerEmbedding:
    @patch("sentence_transformers.SentenceTransformer")
    def test_encode(self, mock_st_cls):
        from flashrag.embeddings.sentence_transformer import SentenceTransformerEmbedding

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.randn(2, 384).astype(np.float32)
        mock_st_cls.return_value = mock_model

        emb = SentenceTransformerEmbedding("all-MiniLM-L6-v2")
        assert emb.dimension == 384
        vecs = emb.encode(["hello", "world"])
        assert vecs.shape == (2, 384)


class TestOpenAIEmbedding:
    @pytest.fixture(autouse=True)
    def _skip_without_openai(self):
        pytest.importorskip("openai")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_encode(self, mock_openai_cls):
        from flashrag.embeddings.openai_embeddings import OpenAIEmbedding

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        fake_data = [MagicMock(embedding=np.random.randn(1536).tolist())]
        mock_client.embeddings.create.return_value = MagicMock(data=fake_data)

        emb = OpenAIEmbedding(api_key="test-key")
        vecs = emb.encode(["test"])
        assert vecs.shape[1] == 1536


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------


class TestPromptTemplates:
    def test_default_template(self):
        from flashrag.generation.prompt_templates import get_template

        tmpl = get_template("default")
        result = tmpl.format(
            question="What is AI?",
            contexts=["AI stands for Artificial Intelligence."],
        )
        assert "What is AI?" in result
        assert "Artificial Intelligence" in result

    def test_format_messages(self):
        from flashrag.generation.prompt_templates import get_template

        tmpl = get_template("default")
        msgs = tmpl.format_messages(
            question="test?",
            contexts=["ctx1"],
        )
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_list_templates(self):
        from flashrag.generation.prompt_templates import list_templates

        templates = list_templates()
        assert "default" in templates
        assert "academic" in templates
        assert "code" in templates

    def test_register_custom(self):
        from flashrag.generation.prompt_templates import (
            PromptTemplate,
            get_template,
            register_template,
        )

        custom = PromptTemplate(name="custom_test", system="sys", user="{context}\n{question}")
        register_template(custom)
        retrieved = get_template("custom_test")
        assert retrieved.name == "custom_test"

    def test_unknown_template_raises(self):
        from flashrag.generation.prompt_templates import get_template

        with pytest.raises(ValueError, match="Unknown template"):
            get_template("nonexistent_xyz")


# ---------------------------------------------------------------------------
# RAGAS Evaluation
# ---------------------------------------------------------------------------


class TestRAGASMetrics:
    def test_context_precision(self):
        from flashrag.analytics.ragas import compute_context_precision

        score = compute_context_precision(
            questions=["What is Python?"],
            contexts=[["Python is a programming language used widely"]],
            ground_truths=["Python is a programming language"],
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.0

    def test_context_recall(self):
        from flashrag.analytics.ragas import compute_context_recall

        score = compute_context_recall(
            questions=["What is Python?"],
            contexts=[["Python is a programming language for data science"]],
            ground_truths=["Python is a programming language"],
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.0

    def test_answer_faithfulness(self):
        from flashrag.analytics.ragas import compute_answer_faithfulness

        score = compute_answer_faithfulness(
            answers=["Python is a programming language used for AI."],
            contexts=[["Python is a programming language. It is used for AI."]],
        )
        assert 0.0 <= score <= 1.0

    def test_answer_relevance(self):
        from flashrag.analytics.ragas import compute_answer_relevance

        score = compute_answer_relevance(
            questions=["What is Python?"],
            answers=["Python is a programming language."],
        )
        assert 0.0 <= score <= 1.0

    def test_evaluator_all_metrics(self):
        from flashrag.analytics.ragas import RAGASEvaluator

        evaluator = RAGASEvaluator()
        results = evaluator.evaluate(
            questions=["What is ML?"],
            answers=["ML is machine learning."],
            contexts=[["Machine learning is a subset of AI."]],
            ground_truths=["Machine learning (ML) is a type of AI."],
        )
        assert "context_precision" in results
        assert "context_recall" in results
        assert "answer_faithfulness" in results
        assert "answer_relevance" in results

    def test_evaluator_single(self):
        from flashrag.analytics.ragas import RAGASEvaluator

        evaluator = RAGASEvaluator(metrics=["answer_relevance"])
        results = evaluator.evaluate_single(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=["Artificial intelligence mimics human cognition."],
        )
        assert "answer_relevance" in results

    def test_empty_inputs(self):
        from flashrag.analytics.ragas import compute_context_precision

        assert compute_context_precision([], [], []) == 0.0


# ---------------------------------------------------------------------------
# Generator (mocked)
# ---------------------------------------------------------------------------


class TestRAGGenerator:
    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForCausalLM")
    def test_generate(self, mock_model_cls, mock_tok_cls):
        import torch

        from flashrag.generation.generator import RAGGenerator

        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"
        mock_tok.pad_token_id = 0
        mock_tok.return_value = {"input_ids": torch.zeros(1, 5, dtype=torch.long)}
        mock_tok.decode.return_value = "Generated answer"
        mock_tok_cls.from_pretrained.return_value = mock_tok

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.generate.return_value = torch.zeros(1, 10, dtype=torch.long)
        mock_model_cls.from_pretrained.return_value = mock_model

        gen = RAGGenerator(model_name="gpt2", device="cpu")
        result = gen.generate(question="What is AI?", contexts=["AI is intelligence."])
        assert result.answer == "Generated answer"
        assert "AI" in result.prompt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestFlashRAGCLI:
    def test_version_cmd(self, capsys):
        import argparse

        from flashrag.cli import _version_cmd

        _version_cmd(argparse.Namespace())
        captured = capsys.readouterr()
        assert "flashrag" in captured.out

    def test_main_no_command(self):
        from flashrag.cli import main

        with pytest.raises(SystemExit):
            with patch("sys.argv", ["flashrag"]):
                main()


# ---------------------------------------------------------------------------
# Integration: load docs → chunk → index → query
# ---------------------------------------------------------------------------


class TestIntegrationRAG:
    def test_full_pipeline(self, tmp_path):
        from flashrag.data.chunkers import RecursiveChunker
        from flashrag.data.loaders import DocumentLoader
        from flashrag.retrieval.bm25 import BM25Retriever
        from flashrag.retrieval.vector_store import VectorStore

        doc_file = tmp_path / "doc.txt"
        doc_file.write_text(
            "Machine learning is a branch of AI. "
            "It enables computers to learn from data. "
            "Deep learning is a subset of machine learning. "
            "Neural networks are the building blocks."
        )

        loader = DocumentLoader()
        docs = loader.load(doc_file)
        assert len(docs) == 1

        chunker = RecursiveChunker(chunk_size=80, chunk_overlap=10)
        chunks = chunker.chunk(docs[0].text)
        assert len(chunks) >= 2

        bm25 = BM25Retriever()
        bm25.index([c.text for c in chunks])

        results = bm25.search("neural networks", top_k=2)
        assert len(results) > 0
        assert any("neural" in r.text.lower() for r in results)

        store = VectorStore(dimension=3)
        rng = np.random.RandomState(42)
        vecs = rng.randn(len(chunks), 3).astype(np.float32)
        store.add(vecs, [c.text for c in chunks])
        assert store.size == len(chunks)
