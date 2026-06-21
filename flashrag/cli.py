"""
FlashRAG command-line interface.

Provides commands for indexing documents, querying, interactive chat,
benchmarking, and system health checks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _version_cmd(args: argparse.Namespace) -> None:
    from flashrag import __version__

    print(f"flashrag {__version__}")


def _settings_cmd(args: argparse.Namespace) -> None:
    import platform

    print("FlashRAG System Settings")
    print("=" * 40)
    print(f"Python     : {platform.python_version()}")
    print(f"Platform   : {platform.platform()}")

    try:
        import torch

        print(f"PyTorch    : {torch.__version__}")
        print(f"CUDA       : {torch.cuda.is_available()} ({torch.version.cuda or 'N/A'})")
        if torch.cuda.is_available():
            print(f"GPU        : {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"GPU Memory : {mem:.1f} GB")
    except ImportError:
        print("PyTorch    : NOT INSTALLED")

    try:
        import faiss

        print(f"FAISS      : {faiss.__version__ if hasattr(faiss, '__version__') else 'installed'}")
    except ImportError:
        print("FAISS      : NOT INSTALLED")

    try:
        import transformers

        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("Transformers: NOT INSTALLED")

    try:
        import sentence_transformers

        print(f"Sentence-T : {sentence_transformers.__version__}")
    except ImportError:
        print("Sentence-T : NOT INSTALLED")


def _check_cmd(args: argparse.Namespace) -> None:
    print("FlashRAG Health Check")
    print("=" * 40)
    checks = {
        "torch": "torch",
        "transformers": "transformers",
        "numpy": "numpy",
        "faiss": "faiss",
        "sentence_transformers": "sentence_transformers",
        "yaml": "yaml",
        "tqdm": "tqdm",
    }
    all_ok = True
    for name, module in checks.items():
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [FAIL] {name}")
            all_ok = False

    optional = {"openai": "openai", "PyPDF2": "PyPDF2", "pdfplumber": "pdfplumber"}
    for name, module in optional.items():
        try:
            __import__(module)
            print(f"  [OK] {name} (optional)")
        except ImportError:
            print(f"  [--] {name} (optional, not installed)")

    print()
    if all_ok:
        print("All required dependencies OK!")
    else:
        print("Some dependencies are missing. Install with: pip install -e '.[all]'")
        sys.exit(1)


def _index_cmd(args: argparse.Namespace) -> None:
    from flashrag.data.preprocessor import Preprocessor
    from flashrag.embeddings.sentence_transformer import SentenceTransformerEmbedding
    from flashrag.retrieval.vector_store import VectorStore

    docs_path = Path(args.docs)
    if not docs_path.exists():
        print(f"Error: path '{docs_path}' does not exist.")
        sys.exit(1)

    print(f"Loading documents from {docs_path}...")
    preprocessor = Preprocessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = preprocessor.process_files([docs_path])
    print(f"  {len(chunks)} chunks created")

    print(f"Encoding with {args.embedding}...")
    embedder = SentenceTransformerEmbedding(
        model_name=args.embedding,
        device=args.device,
    )
    vectors = embedder.encode(
        [c.text for c in chunks],
        show_progress=True,
    )

    store = VectorStore(dimension=embedder.dimension, metric="cosine")
    store.add(vectors, [c.text for c in chunks], [c.metadata for c in chunks])

    output = Path(args.output)
    store.save(output)
    print(f"Index saved to {output} ({store.size} vectors)")


def _query_cmd(args: argparse.Namespace) -> None:
    from flashrag.embeddings.sentence_transformer import SentenceTransformerEmbedding
    from flashrag.retrieval.vector_store import VectorStore
    from flashrag.utils.visualize import print_results

    if not Path(args.index).exists():
        print(f"Error: index '{args.index}' not found. Run 'flashrag index' first.")
        sys.exit(1)

    store = VectorStore.load(args.index)
    embedder = SentenceTransformerEmbedding(model_name=args.embedding, device=args.device)

    query_vec = embedder.encode([args.question])[0]
    results = store.search(query_vec, top_k=args.top_k)

    print(f"\nResults for: '{args.question}'\n")
    print_results(results)


def _chat_cmd(args: argparse.Namespace) -> None:
    from flashrag.embeddings.sentence_transformer import SentenceTransformerEmbedding
    from flashrag.generation.generator import RAGGenerator
    from flashrag.retrieval.vector_store import VectorStore

    store = None
    if args.index and Path(args.index).exists():
        store = VectorStore.load(args.index)

    embedder = SentenceTransformerEmbedding(model_name=args.embedding, device=args.device)
    generator = RAGGenerator(model_name=args.model, device=args.device)

    print("FlashRAG Chat (type 'quit' to exit)")
    print("=" * 40)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not question:
            continue

        contexts = []
        if store:
            query_vec = embedder.encode([question])[0]
            results = store.search(query_vec, top_k=args.top_k)
            contexts = [r.text for r in results]

        if contexts:
            result = generator.generate(question=question, contexts=contexts)
            print(f"\nAssistant: {result.answer}")
            print(f"  (based on {len(contexts)} sources)")
        else:
            answer = generator.generate_simple(question)
            print(f"\nAssistant: {answer}")


def _benchmark_cmd(args: argparse.Namespace) -> None:
    from flashrag.analytics.benchmark import Benchmark
    from flashrag.pipelines.basic_rag import BasicRAGPipeline

    if not args.eval_data:
        print("Error: --eval-data is required for benchmarking.")
        sys.exit(1)

    pipeline = BasicRAGPipeline(
        embedding_model=args.embedding,
        generator_model=args.model,
        top_k=args.top_k,
        device=args.device,
    )

    bench = Benchmark(pipeline=pipeline, output_dir=args.output)
    results = bench.run(eval_path=args.eval_data)

    print("\nBenchmark Results:")
    print("=" * 40)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="flashrag",
        description="FlashRAG — Retrieval-Augmented Generation CLI",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="Show version")
    sub.add_parser("settings", help="Show system settings")
    sub.add_parser("check", help="Run health check")

    idx_p = sub.add_parser("index", help="Index documents")
    idx_p.add_argument("--docs", required=True, help="Path to documents")
    idx_p.add_argument("--embedding", default="all-MiniLM-L6-v2")
    idx_p.add_argument("--output", default="workspace/index")
    idx_p.add_argument("--chunk-size", type=int, default=512)
    idx_p.add_argument("--chunk-overlap", type=int, default=64)
    idx_p.add_argument("--device", default="cpu")

    q_p = sub.add_parser("query", help="Query the index")
    q_p.add_argument("--question", required=True, help="Query string")
    q_p.add_argument("--index", default="workspace/index")
    q_p.add_argument("--embedding", default="all-MiniLM-L6-v2")
    q_p.add_argument("--top-k", type=int, default=5)
    q_p.add_argument("--device", default="cpu")

    chat_p = sub.add_parser("chat", help="Interactive RAG chat")
    chat_p.add_argument("--model", default="gpt2")
    chat_p.add_argument("--index", default=None)
    chat_p.add_argument("--embedding", default="all-MiniLM-L6-v2")
    chat_p.add_argument("--top-k", type=int, default=5)
    chat_p.add_argument("--device", default="cpu")

    bench_p = sub.add_parser("benchmark", help="Benchmark RAG pipeline")
    bench_p.add_argument("--eval-data", required=True, help="Path to eval JSONL")
    bench_p.add_argument("--embedding", default="all-MiniLM-L6-v2")
    bench_p.add_argument("--model", default="gpt2")
    bench_p.add_argument("--top-k", type=int, default=5)
    bench_p.add_argument("--output", default="workspace/benchmark")
    bench_p.add_argument("--device", default="cpu")

    args = parser.parse_args()

    cmd_map = {
        "version": _version_cmd,
        "settings": _settings_cmd,
        "check": _check_cmd,
        "index": _index_cmd,
        "query": _query_cmd,
        "chat": _chat_cmd,
        "benchmark": _benchmark_cmd,
    }

    if args.command in cmd_map:
        cmd_map[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
