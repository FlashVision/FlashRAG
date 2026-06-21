"""
Basic RAG Example — Retrieve → Generate

Demonstrates the standard RAG pipeline: index documents, retrieve
relevant passages for a query, and generate a grounded answer.
"""

from flashrag import BasicRAGPipeline

SAMPLE_DOCS = [
    "The transformer architecture was introduced in the paper 'Attention Is All You Need' "
    "by Vaswani et al. in 2017. It uses self-attention mechanisms to process sequences "
    "in parallel, replacing recurrent layers entirely.",
    "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained "
    "language model developed by Google in 2018. It uses masked language modeling and "
    "next sentence prediction as pre-training objectives.",
    "GPT (Generative Pre-trained Transformer) by OpenAI uses autoregressive language "
    "modeling. GPT-2 demonstrated that scaling up model size and training data leads "
    "to significant improvements in text generation quality.",
    "Retrieval-Augmented Generation (RAG) combines a retriever with a generator model. "
    "The retriever fetches relevant documents from a corpus, and the generator produces "
    "an answer conditioned on both the query and retrieved contexts.",
    "Vector databases store high-dimensional embeddings and enable fast similarity search. "
    "FAISS by Meta supports both exact and approximate nearest neighbor search with "
    "support for billions of vectors.",
    "Sentence transformers produce fixed-length dense vector representations of sentences. "
    "Models like all-MiniLM-L6-v2 map sentences to a 384-dimensional space where "
    "semantically similar sentences are close together.",
]


def main():
    print("=" * 60)
    print("FlashRAG — Basic RAG Example")
    print("=" * 60)

    pipeline = BasicRAGPipeline(
        embedding_model="all-MiniLM-L6-v2",
        generator_model="gpt2",
        top_k=3,
        device="cpu",
    )

    print(f"\nIndexing {len(SAMPLE_DOCS)} documents...")
    pipeline.index_documents(texts=SAMPLE_DOCS)

    questions = [
        "What is the transformer architecture?",
        "How does RAG work?",
        "What is FAISS used for?",
    ]

    for question in questions:
        print(f"\n{'─' * 40}")
        print(f"Q: {question}")
        result = pipeline.run(question)
        print(f"A: {result.answer}")
        print(f"   Sources: {len(result.contexts)} contexts, top score: {result.scores[0]:.4f}")


if __name__ == "__main__":
    main()
