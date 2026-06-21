# Quick Start

## Document QA in 5 Lines

```python
from flashrag import DocumentQA

qa = DocumentQA(embedding_model="all-MiniLM-L6-v2", generator_model="gpt2")
qa.add_documents(["paper.pdf", "notes.md"])
answer = qa.ask("What is the main finding?")
print(answer)
```

## Basic RAG Pipeline

```python
from flashrag import BasicRAGPipeline

pipeline = BasicRAGPipeline(
    embedding_model="all-MiniLM-L6-v2",
    generator_model="gpt2",
    top_k=5,
)

pipeline.index_documents(texts=[
    "Transformers use self-attention mechanisms...",
    "BERT uses masked language modeling...",
    "RAG combines retrieval with generation...",
])

result = pipeline.run("What is a transformer?")
print(result.answer)
print(result.sources)
```

## CLI Usage

```bash
# Index documents
flashrag index --docs ./my_docs/ --embedding all-MiniLM-L6-v2

# Query
flashrag query --question "What is attention?" --top-k 5

# Interactive chat
flashrag chat --model gpt2 --index workspace/index
```

## Using Config Files

```bash
flashrag index --docs ./papers/ --config configs/flashrag_basic.yaml
```

```python
from flashrag.cfg import get_config

config = get_config("configs/flashrag_hybrid_search.yaml")
```
