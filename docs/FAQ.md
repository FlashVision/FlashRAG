# FAQ

## General

**Q: What models does FlashRAG support?**
A: Any HuggingFace causal LM for generation (GPT-2, LLaMA, Mistral, Phi, etc.) and any SentenceTransformer model for embeddings. OpenAI API and CLIP are also supported.

**Q: Do I need a GPU?**
A: No. FlashRAG works on CPU. A GPU will significantly speed up embedding and generation.

**Q: What vector database does FlashRAG use?**
A: FAISS (Facebook AI Similarity Search) for dense vector retrieval, with a pure-Python BM25 implementation for sparse retrieval.

## Installation

**Q: I get `ImportError: faiss`**
A: Install FAISS with `pip install faiss-cpu` (or `faiss-gpu` for GPU support).

**Q: I get `ImportError: sentence_transformers`**
A: Install with `pip install sentence-transformers`.

**Q: How do I use OpenAI embeddings?**
A: Install with `pip install 'flashrag[openai]'` and set `OPENAI_API_KEY`.

## Usage

**Q: How do I index PDF files?**
A: Install `pip install 'flashrag[pdf]'` then use `DocumentQA.add_documents(["file.pdf"])`.

**Q: Can I mix text and image retrieval?**
A: Yes, use the `MultimodalRAGPipeline` with CLIP embeddings.

**Q: How do I improve retrieval quality?**
A: Try hybrid search (`HybridSearch`) and cross-encoder reranking (`CrossEncoderReranker`).

## Troubleshooting

**Q: Low retrieval scores?**
A: Check chunk size (try 256–1024), ensure documents are preprocessed, consider hybrid search.

**Q: Out of memory?**
A: Reduce batch size, use a smaller embedding model, or use CPU offloading.
