"""
Hybrid Search Example — Dense + Sparse Retrieval

Demonstrates combining FAISS dense search with BM25 sparse search
using reciprocal rank fusion for improved retrieval quality.
"""

from flashrag.retrieval import HybridSearch

CORPUS = [
    "Python is a high-level programming language known for its readability and "
    "versatility. It supports multiple programming paradigms including procedural, "
    "object-oriented, and functional programming.",
    "JavaScript is the language of the web. It runs in browsers and on servers "
    "via Node.js. Modern JavaScript uses ES6+ features like arrow functions, "
    "destructuring, and async/await.",
    "Rust is a systems programming language focused on safety, speed, and "
    "concurrency. Its ownership system prevents data races at compile time "
    "without needing a garbage collector.",
    "Go (Golang) was developed at Google. It features built-in concurrency with "
    "goroutines and channels, a simple type system, and fast compilation times.",
    "PyTorch is an open-source machine learning framework developed by Meta AI. "
    "It provides dynamic computational graphs and is widely used in research "
    "for deep learning experiments.",
    "TensorFlow is Google's machine learning framework that supports both eager "
    "and graph execution modes. TensorFlow 2.x uses Keras as its high-level API "
    "for building neural networks.",
]


def main():
    print("=" * 60)
    print("FlashRAG — Hybrid Search Example")
    print("=" * 60)

    hybrid = HybridSearch(
        embedding_model="all-MiniLM-L6-v2",
        alpha=0.6,
    )

    print(f"\nIndexing {len(CORPUS)} documents...")
    hybrid.index(CORPUS)

    queries = [
        "machine learning framework for research",
        "systems programming with memory safety",
        "web development language",
    ]

    for query in queries:
        print(f"\n{'─' * 40}")
        print(f"Query: {query}")
        results = hybrid.search(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] (score={r.score:.4f}) {r.text[:100]}...")


if __name__ == "__main__":
    main()
