# Embeddings

FlashRAG supports multiple embedding backends.

## SentenceTransformer

Dense text embeddings using the `sentence-transformers` library.

```python
from flashrag.embeddings import SentenceTransformerEmbedding

embed = SentenceTransformerEmbedding("all-MiniLM-L6-v2", device="cuda")
vectors = embed.encode(["Hello world", "FlashRAG is great"])
```

### Recommended Models

| Model | Dimension | Speed | Quality |
|-------|-----------|-------|---------|
| all-MiniLM-L6-v2 | 384 | Fast | Good |
| all-mpnet-base-v2 | 768 | Medium | Better |
| bge-large-en-v1.5 | 1024 | Slower | Best |

## OpenAI Embeddings

API-based embeddings via OpenAI.

```python
from flashrag.embeddings import OpenAIEmbedding

embed = OpenAIEmbedding("text-embedding-3-small")
vectors = embed.encode(["Hello world"])
```

Requires `OPENAI_API_KEY` environment variable.

## Vision Embeddings (CLIP/SigLIP)

Multimodal embeddings for text and images.

```python
from flashrag.embeddings import VisionEmbedding

embed = VisionEmbedding("openai/clip-vit-base-patch32")
text_vecs = embed.encode(["a cat on a mat"])
image_vecs = embed.encode_images(["photo.jpg"])
```

## Custom Embeddings

Implement the `BaseEmbedding` interface:

```python
from flashrag.embeddings import BaseEmbedding

class MyEmbedding(BaseEmbedding):
    @property
    def dimension(self) -> int:
        return 256

    def encode(self, texts, batch_size=64, show_progress=False, normalize=True):
        # Your implementation
        pass
```
