# Installation

## Requirements

- Python 3.10+
- PyTorch 2.1+

## pip install

```bash
pip install flashrag

# With all optional extras
pip install "flashrag[all]"
```

## From source

```bash
git clone https://github.com/Gaurav14cs17/FlashVision.git
cd FlashVision/FlashRAG
pip install -e ".[all,dev]"
```

## Optional Extras

| Extra | Packages | Use Case |
|-------|----------|----------|
| `openai` | openai | OpenAI API embeddings and generation |
| `pdf` | PyPDF2, pdfplumber | PDF document loading |
| `vision` | Pillow, open-clip-torch | CLIP/SigLIP image embeddings |
| `all` | All of the above | Everything |
| `dev` | pytest, ruff, mypy | Development tools |

## Verify

```bash
flashrag check      # health check
flashrag settings   # system info
flashrag version    # version
```

## Environment Setup (conda)

```bash
bash setup_env.sh
```
