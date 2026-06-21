# FlashRAG Documentation

Welcome to the FlashRAG documentation — a production-grade Retrieval-Augmented Generation framework.

## Overview

FlashRAG provides an end-to-end pipeline for building RAG systems:

1. **Document Loading** — PDF, HTML, Markdown, CSV, plain text
2. **Chunking** — Recursive, sentence-level, and fixed-size strategies
3. **Embedding** — SentenceTransformer, OpenAI, CLIP/SigLIP
4. **Retrieval** — FAISS vector search, BM25, hybrid dense+sparse, cross-encoder reranking
5. **Generation** — LLM-based answer generation with context injection
6. **Evaluation** — Recall@K, MRR, NDCG, faithfulness, relevance

## Quick Links

- [Installation](Installation.md)
- [Quick Start](Quick-Start.md)
- [Retrieval](Retrieval.md)
- [Embeddings](Embeddings.md)
- [Generation](Generation.md)
- [Evaluation](Evaluation.md)
- [FAQ](FAQ.md)

## Architecture

```
Document → Loader → Chunker → Embedder → Vector Store
                                              ↓
Query → Embedder → Retriever → (Reranker) → Generator → Answer
```
