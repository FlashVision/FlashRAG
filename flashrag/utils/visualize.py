"""
Visualization and display utilities.
"""

from __future__ import annotations

from typing import Any

from flashrag.retrieval.vector_store import SearchResult


def format_search_results(
    results: list[SearchResult],
    max_text_len: int = 200,
    show_metadata: bool = True,
) -> str:
    """Format search results for display."""
    if not results:
        return "No results found."

    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        text_preview = r.text[:max_text_len]
        if len(r.text) > max_text_len:
            text_preview += "..."

        lines.append(f"[{i}] Score: {r.score:.4f}")
        lines.append(f"    {text_preview}")

        if show_metadata and r.metadata:
            meta_items = []
            for k, v in r.metadata.items():
                if k not in ("chunk_index",):
                    meta_items.append(f"{k}={v}")
            if meta_items:
                lines.append(f"    Meta: {', '.join(meta_items)}")
        lines.append("")

    return "\n".join(lines)


def print_results(
    results: list[SearchResult],
    max_text_len: int = 200,
    show_metadata: bool = True,
) -> None:
    """Print formatted search results."""
    print(format_search_results(results, max_text_len, show_metadata))


def format_rag_result(result: Any, show_contexts: bool = True) -> str:
    """Format a RAGResult for display."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("ANSWER:")
    lines.append(result.answer)
    lines.append("")

    if show_contexts and result.contexts:
        lines.append(f"SOURCES ({len(result.contexts)} contexts):")
        for i, ctx in enumerate(result.contexts, start=1):
            preview = ctx[:150] + "..." if len(ctx) > 150 else ctx
            score = result.scores[i - 1] if i - 1 < len(result.scores) else 0
            lines.append(f"  [{i}] (score={score:.4f}) {preview}")
        lines.append("")

    if result.citations:
        lines.append(
            f"CITATIONS: {len(result.citations.citations)} found, "
            f"attribution={result.citations.attribution_score:.2f}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)
