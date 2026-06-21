"""
RAG Benchmark Example — Evaluate Retrieval and Generation Quality

Demonstrates how to benchmark a RAG pipeline using standard metrics:
Recall@K, MRR, NDCG, faithfulness, and relevance.
"""

from flashrag.analytics import Benchmark, compute_recall_at_k, compute_mrr, compute_ndcg
from flashrag.analytics.metrics import compute_faithfulness, compute_relevance
from flashrag.pipelines import BasicRAGPipeline

CORPUS = [
    "Machine learning is a subset of artificial intelligence that enables systems "
    "to learn from data without being explicitly programmed.",
    "Deep learning uses neural networks with many layers to learn representations "
    "of data with multiple levels of abstraction.",
    "Reinforcement learning is a type of machine learning where an agent learns "
    "to make decisions by interacting with an environment.",
    "Natural language processing (NLP) deals with the interaction between "
    "computers and human language, enabling machines to understand text.",
    "Computer vision enables machines to interpret and understand visual "
    "information from images and videos.",
    "Transfer learning allows a model trained on one task to be adapted for "
    "a different but related task, reducing the need for large datasets.",
]

EVAL_DATA = [
    {
        "question": "What is machine learning?",
        "answer": "Machine learning is a subset of AI that enables systems to learn from data.",
        "relevant_docs": [CORPUS[0]],
    },
    {
        "question": "How does deep learning work?",
        "answer": "Deep learning uses multi-layer neural networks to learn data representations.",
        "relevant_docs": [CORPUS[1]],
    },
    {
        "question": "What is transfer learning?",
        "answer": "Transfer learning adapts a model trained on one task for a different task.",
        "relevant_docs": [CORPUS[5]],
    },
]


def main():
    print("=" * 60)
    print("FlashRAG — Benchmark Example")
    print("=" * 60)

    print("\n1. Running standalone metric computations...")

    retrieved = [[CORPUS[0], CORPUS[1]], [CORPUS[1], CORPUS[0]], [CORPUS[5], CORPUS[3]]]
    relevant = [[CORPUS[0]], [CORPUS[1]], [CORPUS[5]]]

    print(f"   Recall@1 = {compute_recall_at_k(retrieved, relevant, k=1):.4f}")
    print(f"   Recall@2 = {compute_recall_at_k(retrieved, relevant, k=2):.4f}")
    print(f"   MRR      = {compute_mrr(retrieved, relevant):.4f}")
    print(f"   NDCG@2   = {compute_ndcg(retrieved, relevant, k=2):.4f}")

    answer = "Machine learning enables systems to learn patterns from data."
    contexts = [CORPUS[0], CORPUS[1]]
    print(f"\n   Faithfulness = {compute_faithfulness(answer, contexts):.4f}")
    print(f"   Relevance    = {compute_relevance(answer, 'What is machine learning?'):.4f}")

    print("\n2. Building RAG pipeline for full benchmark...")
    pipeline = BasicRAGPipeline(
        embedding_model="all-MiniLM-L6-v2",
        generator_model="gpt2",
        top_k=3,
        device="cpu",
    )
    pipeline.index_documents(texts=CORPUS)

    bench = Benchmark(pipeline=pipeline, output_dir="workspace/benchmark")
    results = bench.run(eval_data=EVAL_DATA, ks=[1, 3, 5])

    print("\nBenchmark Results:")
    print("=" * 40)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
