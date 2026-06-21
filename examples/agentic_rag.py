"""
Agentic RAG Example — Adaptive Retrieval with Query Decomposition

Demonstrates the agentic RAG pipeline that decomposes complex queries
into sub-questions and performs multi-step retrieval.
"""

from flashrag.pipelines import AgenticRAGPipeline

KNOWLEDGE_BASE = [
    "TCP (Transmission Control Protocol) is a connection-oriented protocol that "
    "provides reliable, ordered delivery of data. It uses a three-way handshake "
    "for connection establishment and flow control mechanisms.",
    "UDP (User Datagram Protocol) is a connectionless protocol that provides "
    "fast but unreliable delivery. It has minimal overhead, no handshaking, "
    "and is used for real-time applications like video streaming and gaming.",
    "HTTP/2 improves upon HTTP/1.1 by introducing multiplexing, header "
    "compression, and server push. It allows multiple requests over a single "
    "TCP connection.",
    "WebSocket provides full-duplex communication over a single TCP connection. "
    "Unlike HTTP, it allows the server to push data to the client without "
    "the client requesting it.",
    "DNS (Domain Name System) translates human-readable domain names to IP "
    "addresses. It uses a hierarchical distributed naming system with root "
    "servers, TLD servers, and authoritative name servers.",
    "TLS (Transport Layer Security) provides encryption for data in transit. "
    "It uses asymmetric cryptography for key exchange and symmetric encryption "
    "for data transfer, with certificates for authentication.",
]


def main():
    print("=" * 60)
    print("FlashRAG — Agentic RAG Example")
    print("=" * 60)

    pipeline = AgenticRAGPipeline(
        embedding_model="all-MiniLM-L6-v2",
        generator_model="gpt2",
        top_k=3,
        max_steps=3,
        device="cpu",
    )

    print(f"\nIndexing {len(KNOWLEDGE_BASE)} documents...")
    pipeline.index_documents(KNOWLEDGE_BASE)

    question = "Compare TCP and UDP protocols and explain when to use each"
    print(f"\n{'─' * 40}")
    print(f"Complex Query: {question}")

    sub_queries = pipeline.decompose_query(question)
    print(f"Decomposed into {len(sub_queries)} sub-queries:")
    for i, sq in enumerate(sub_queries, 1):
        print(f"  {i}. {sq}")

    result = pipeline.run(question)
    print(f"\nAnswer: {result.answer}")
    print(f"Sources used: {len(result.contexts)}")
    print(f"Agent steps: {result.metadata.get('agent_steps', 'N/A')}")


if __name__ == "__main__":
    main()
