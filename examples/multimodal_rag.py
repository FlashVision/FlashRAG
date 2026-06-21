"""
Multimodal RAG Example — Text + Image Retrieval

Demonstrates cross-modal retrieval using CLIP embeddings to search
both text and images with a text query.
"""

from flashrag.pipelines import MultimodalRAGPipeline

SAMPLE_TEXTS = [
    "A golden retriever playing fetch in a sunny park with green grass.",
    "The Eiffel Tower illuminated at night with the city lights of Paris below.",
    "A diagram showing the transformer architecture with attention heads.",
    "A cat sleeping on a warm laptop keyboard in a cozy home office.",
    "Neural network training curves showing loss decreasing over epochs.",
]


def main():
    print("=" * 60)
    print("FlashRAG — Multimodal RAG Example")
    print("=" * 60)

    pipeline = MultimodalRAGPipeline(
        vision_model="openai/clip-vit-base-patch32",
        generator_model="gpt2",
        top_k=3,
        device="cpu",
    )

    print(f"\nIndexing {len(SAMPLE_TEXTS)} text descriptions...")
    pipeline.index_texts(SAMPLE_TEXTS)

    queries = [
        "dog playing outside",
        "famous landmark in France",
        "machine learning visualization",
    ]

    for query in queries:
        print(f"\n{'─' * 40}")
        print(f"Query: {query}")
        results = pipeline.search_by_text(query, top_k=2)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] (score={r.score:.4f}) {r.text}")


if __name__ == "__main__":
    main()
