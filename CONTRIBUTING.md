# Contributing to FlashRAG

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
git clone https://github.com/Gaurav14cs17/FlashVision.git
cd FlashVision/FlashRAG
pip install -e ".[all,dev]"
pre-commit install
```

## Code Style

- We use **ruff** for linting and formatting.
- Line length: 100 characters.
- Type hints on all public functions.

```bash
ruff check flashrag/
ruff format flashrag/
```

## Testing

```bash
pytest tests/ -v
flashrag check
```

## Pull Requests

1. Fork the repo and create your branch from `main`.
2. Add tests for any new functionality.
3. Ensure all tests pass and linting is clean.
4. Write clear commit messages.
5. Open a PR with a description of your changes.

## Reporting Issues

- Use GitHub Issues.
- Include steps to reproduce, expected vs. actual behavior, and environment info.
