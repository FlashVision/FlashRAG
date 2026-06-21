# Generation

## RAG Generator

Generate answers grounded in retrieved context using HuggingFace LLMs.

```python
from flashrag.generation import RAGGenerator

gen = RAGGenerator(model_name="gpt2", device="cuda")
result = gen.generate(
    question="What is attention?",
    contexts=["Attention is a mechanism...", "Self-attention computes..."],
)
print(result.answer)
```

## Prompt Templates

FlashRAG includes several built-in templates:

| Template | Use Case |
|----------|----------|
| `default` | General-purpose QA with citations |
| `conversational` | Friendly, natural tone |
| `academic` | Formal with in-text citations |
| `code` | Programming-related QA |
| `minimal` | Bare-bones context + question |

```python
from flashrag.generation import get_template

template = get_template("academic")
prompt = template.format(
    question="What is RAG?",
    contexts=["RAG combines retrieval..."],
    sources=["paper.pdf"],
)
```

### Custom Templates

```python
from flashrag.generation import PromptTemplate, register_template

my_template = PromptTemplate(
    name="custom",
    system="You are a domain expert...",
    user="References:\n{context}\n\nQ: {question}\nA:",
)
register_template(my_template)
```

## Citation Extraction

Extract and validate inline citations from generated answers.

```python
from flashrag.generation import CitationExtractor

extractor = CitationExtractor()
report = extractor.extract(answer, contexts)
print(report.cited_sources)
print(report.attribution_score)
```
