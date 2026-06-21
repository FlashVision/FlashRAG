# Evaluation

## Retrieval Metrics

### Recall@K

Fraction of relevant documents found in the top-K results.

```python
from flashrag.analytics import compute_recall_at_k

recall = compute_recall_at_k(retrieved, relevant, k=5)
```

### MRR (Mean Reciprocal Rank)

Average of 1/rank of the first relevant result.

```python
from flashrag.analytics import compute_mrr

mrr = compute_mrr(retrieved, relevant)
```

### NDCG@K (Normalized Discounted Cumulative Gain)

Measures ranking quality with position-weighted relevance.

```python
from flashrag.analytics import compute_ndcg

ndcg = compute_ndcg(retrieved, relevant, k=10)
```

## Generation Metrics

### Faithfulness

How well the answer is grounded in the provided contexts.

```python
from flashrag.analytics import compute_faithfulness

score = compute_faithfulness(answer, contexts)
```

### Relevance

How well the answer addresses the question.

```python
from flashrag.analytics import compute_relevance

score = compute_relevance(answer, question)
```

## Benchmark Suite

Run a comprehensive evaluation on your RAG pipeline.

```python
from flashrag.analytics import Benchmark

bench = Benchmark(pipeline=my_pipeline)
results = bench.run(eval_data=eval_dataset, ks=[1, 3, 5, 10])
```

### Evaluation Data Format (JSONL)

```json
{"question": "What is RAG?", "answer": "RAG combines...", "relevant_docs": ["doc text..."]}
```
