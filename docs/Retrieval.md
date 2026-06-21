# Retrieval

FlashRAG supports multiple retrieval strategies.

## Vector Store (Dense Retrieval)

FAISS-backed vector search with cosine similarity or L2 distance.

```python
from flashrag.retrieval import VectorStore

store = VectorStore(dimension=384, metric="cosine")
store.add(vectors, documents, metadata)
results = store.search(query_vector, top_k=10)
```

### Save and Load

```python
store.save("my_index/")
store = VectorStore.load("my_index/")
```

## BM25 (Sparse Retrieval)

Okapi BM25 with TF-IDF scoring for keyword-based search.

```python
from flashrag.retrieval import BM25Retriever

bm25 = BM25Retriever(k1=1.5, b=0.75)
bm25.index(documents)
results = bm25.search("query text", top_k=10)
```

## Hybrid Search

Combines dense and sparse retrieval with Reciprocal Rank Fusion.

```python
from flashrag.retrieval import HybridSearch

hybrid = HybridSearch(embedding_model="all-MiniLM-L6-v2", alpha=0.5)
hybrid.index(documents)
results = hybrid.search("query", top_k=10)
```

### Alpha Parameter

- `alpha=1.0` → pure dense retrieval
- `alpha=0.0` → pure BM25
- `alpha=0.5` → equal weight (recommended starting point)

## Cross-Encoder Reranking

Re-score initial retrieval results with a cross-encoder for higher precision.

```python
from flashrag.retrieval import CrossEncoderReranker

reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranked = reranker.rerank(query, initial_results, top_k=3)
```
