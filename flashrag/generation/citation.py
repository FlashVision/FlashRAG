"""
Source citation and attribution extraction.

Parses generated answers for inline citations like [1], [2] and maps
them back to the source documents. Also computes attribution scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Citation:
    citation_id: int
    text_span: str
    source_text: str
    source_metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class CitationReport:
    citations: list[Citation]
    cited_sources: list[int]
    uncited_sources: list[int]
    attribution_score: float
    answer_with_highlights: str


class CitationExtractor:
    """
    Extract and validate inline citations from generated answers.

    Looks for patterns like ``[1]``, ``[2]``, ``[1, 3]`` in the answer
    and maps them back to the provided context passages.
    """

    _CITE_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

    def extract(
        self,
        answer: str,
        contexts: list[str],
        sources: list[dict[str, Any]] | None = None,
    ) -> CitationReport:
        """
        Extract citations from the answer and build a report.

        Parameters
        ----------
        answer : str
            The generated answer text with inline citations.
        contexts : list of str
            The context passages that were provided to the generator.
        sources : list of dict, optional
            Metadata for each context passage.
        """
        sources = sources or [{}] * len(contexts)
        citations: list[Citation] = []
        cited_ids: set = set()

        for match in self._CITE_PATTERN.finditer(answer):
            raw_ids = match.group(1)
            ids = [int(x.strip()) for x in raw_ids.split(",")]

            span_start = max(0, match.start() - 100)
            span_end = min(len(answer), match.end() + 50)
            text_span = answer[span_start:span_end].strip()

            for cid in ids:
                idx = cid - 1
                if 0 <= idx < len(contexts):
                    cited_ids.add(cid)
                    confidence = self._compute_overlap(text_span, contexts[idx])
                    citations.append(
                        Citation(
                            citation_id=cid,
                            text_span=text_span,
                            source_text=contexts[idx][:200],
                            source_metadata=sources[idx] if idx < len(sources) else {},
                            confidence=confidence,
                        )
                    )

        all_ids = set(range(1, len(contexts) + 1))
        uncited = sorted(all_ids - cited_ids)

        attribution = len(cited_ids) / len(contexts) if contexts else 0.0

        highlighted = self._highlight_citations(answer)

        return CitationReport(
            citations=citations,
            cited_sources=sorted(cited_ids),
            uncited_sources=uncited,
            attribution_score=attribution,
            answer_with_highlights=highlighted,
        )

    def _compute_overlap(self, span: str, source: str) -> float:
        """Estimate how well the answer span is supported by the source text."""
        span_words = set(span.lower().split())
        source_words = set(source.lower().split())
        if not span_words:
            return 0.0
        overlap = span_words & source_words
        return len(overlap) / len(span_words)

    def _highlight_citations(self, answer: str) -> str:
        """Wrap citations in bold markers for display."""

        def replacer(match: re.Match) -> str:
            return f"**{match.group(0)}**"

        return self._CITE_PATTERN.sub(replacer, answer)

    def validate_citations(
        self,
        answer: str,
        contexts: list[str],
        threshold: float = 0.3,
    ) -> dict[str, Any]:
        """
        Validate that citations in the answer are actually supported by the sources.

        Returns a dict with valid/invalid citations and overall fidelity score.
        """
        report = self.extract(answer, contexts)
        valid = [c for c in report.citations if c.confidence >= threshold]
        invalid = [c for c in report.citations if c.confidence < threshold]

        fidelity = len(valid) / len(report.citations) if report.citations else 1.0

        return {
            "valid_citations": valid,
            "invalid_citations": invalid,
            "fidelity_score": fidelity,
            "total_citations": len(report.citations),
            "attribution_score": report.attribution_score,
        }
