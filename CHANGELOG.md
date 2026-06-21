# Changelog

All notable changes to FlashRAG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-01

### Added

- FAISS-backed vector store with add / search / save / load
- BM25 sparse retrieval with TF-IDF scoring
- Hybrid dense + sparse search with reciprocal-rank fusion
- Cross-encoder reranking via HuggingFace models
- SentenceTransformer, OpenAI, and CLIP/SigLIP embedding backends
- Document loaders for PDF, HTML, Markdown, CSV
- Recursive and semantic text chunking
- LLM-based answer generation with context injection
- RAG prompt templates (default, conversational, academic, code)
- Source citation and attribution extraction
- BasicRAG, AgenticRAG, MultimodalRAG, CorrectiveRAG pipelines
- DocumentQA, KnowledgeBase, ResearchAssistant solutions
- Retrieval metrics: Recall@K, MRR, NDCG, faithfulness, relevance
- CLI with `index`, `query`, `chat`, `benchmark` commands
- Docker support with Dockerfile and docker-compose
- CI with GitHub Actions
