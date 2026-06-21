"""
Query transformation strategies for retrieval augmentation.

Provides rule-based query decomposition, step-back prompting,
multi-query generation, and intelligent query routing.  These
transformations improve retrieval recall by reformulating or
expanding queries before they reach the retriever.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from flashrag.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)

_CONJUNCTIONS = re.compile(
    r"\b(?:and also|and then|and|as well as|in addition to)\b",
    re.IGNORECASE,
)
_QUESTION_WORDS = frozenset(
    {
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "which",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "should",
    }
)

_SYNONYM_MAP: dict[str, list[str]] = {
    "best": ["top", "optimal", "most effective"],
    "way": ["method", "approach", "technique"],
    "use": ["utilize", "employ", "apply"],
    "make": ["create", "build", "construct"],
    "big": ["large", "significant", "substantial"],
    "small": ["tiny", "minor", "compact"],
    "fast": ["quick", "rapid", "speedy"],
    "good": ["effective", "excellent", "high-quality"],
    "bad": ["poor", "ineffective", "suboptimal"],
    "important": ["critical", "essential", "crucial"],
    "different": ["various", "distinct", "diverse"],
    "show": ["demonstrate", "illustrate", "display"],
    "help": ["assist", "support", "aid"],
    "find": ["discover", "locate", "identify"],
    "change": ["modify", "alter", "adjust"],
    "improve": ["enhance", "optimize", "boost"],
    "problem": ["issue", "challenge", "difficulty"],
    "example": ["instance", "case", "illustration"],
}


class QueryDecomposer:
    """
    Break complex queries into simpler sub-queries.

    Supports rule-based decomposition that splits on conjunctions
    and identifies multiple question aspects within a single query.

    Parameters
    ----------
    method : str
        Decomposition strategy – currently ``"rule_based"`` is supported.
    """

    def __init__(self, method: str = "rule_based") -> None:
        if method not in ("rule_based",):
            raise ValueError(f"Unknown decomposition method '{method}'. Supported: 'rule_based'")
        self.method = method

    def decompose(self, query: str) -> list[str]:
        """
        Decompose *query* into a list of sub-queries.

        Parameters
        ----------
        query : str
            The complex query to decompose.

        Returns
        -------
        list[str]
            One or more simpler sub-queries.
        """
        if self.method == "rule_based":
            return self._rule_based_decompose(query)
        return [query]

    @staticmethod
    def _rule_based_decompose(query: str) -> list[str]:
        """Split a query on conjunctions and question boundaries."""
        parts = _CONJUNCTIONS.split(query)
        sub_queries: list[str] = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            sentences = re.split(r"[.?!;]\s*", part)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 5:
                    continue

                if not sent.endswith("?"):
                    words = sent.lower().split()
                    if words and words[0] in _QUESTION_WORDS:
                        sent = sent.rstrip(".") + "?"

                sub_queries.append(sent)

        if not sub_queries:
            sub_queries = [query]

        logger.debug(
            "Decomposed query into %d sub-queries",
            len(sub_queries),
        )
        return sub_queries


class StepBackPrompter:
    """
    Generate a more abstract "step-back" version of a query.

    The step-back prompt captures the broader topic or principle
    behind a specific question, which can help retrievers find
    foundational context that a narrow query might miss.
    """

    _SPECIFICITY_PATTERNS = [
        (re.compile(r"\bin (\d{4})\b", re.I), ""),
        (re.compile(r"\bon (\w+ \d{1,2},? \d{4})\b", re.I), ""),
        (re.compile(r"\bspecifically\b", re.I), ""),
        (re.compile(r"\bexactly\b", re.I), ""),
    ]

    _GENERALISATION_RULES: list[tuple[re.Pattern[str], str]] = [
        (
            re.compile(
                r"^(what|how)\b.*?\b(specific|particular)\b",
                re.I,
            ),
            "general",
        ),
        (
            re.compile(r"^why did (.+?) (in|on|at|during)\b", re.I),
            "why does {0}",
        ),
    ]

    def generate_stepback(self, query: str) -> str:
        """
        Create a more general version of *query*.

        Parameters
        ----------
        query : str
            The specific query to generalise.

        Returns
        -------
        str
            A broader version of the query.
        """
        stepped = query

        for pattern, replacement in self._SPECIFICITY_PATTERNS:
            stepped = pattern.sub(replacement, stepped)
        stepped = re.sub(r"\s{2,}", " ", stepped).strip()

        for pattern, template in self._GENERALISATION_RULES:
            match = pattern.search(stepped)
            if match:
                groups = match.groups()
                if "{0}" in template:
                    stepped = template.format(*groups)
                else:
                    stepped = pattern.sub(template, stepped)
                break

        stepped = self._extract_core_concepts(stepped)

        if stepped == query:
            stepped = self._broaden_with_prefix(query)

        logger.debug("Step-back: '%s' → '%s'", query, stepped)
        return stepped

    @staticmethod
    def _extract_core_concepts(query: str) -> str:
        """Remove leading question scaffolding to extract key concepts."""
        prefixes = [
            r"^can you (?:tell me |explain )?",
            r"^(?:please )?(?:tell me |explain )",
            r"^I (?:want|need) to (?:know|understand) ",
        ]
        result = query
        for prefix in prefixes:
            result = re.sub(prefix, "", result, flags=re.I)
        return result.strip() or query

    @staticmethod
    def _broaden_with_prefix(query: str) -> str:
        """Add a generalising prefix when other strategies don't apply."""
        query_clean = query.rstrip("?").strip()
        return f"What are the general principles behind {query_clean}?"


class MultiQueryGenerator:
    """
    Generate multiple reformulations of a query.

    Produces alternative phrasings through synonym expansion,
    structural rephrasing, and perspective shifts to improve
    retrieval recall.

    Parameters
    ----------
    num_queries : int
        Number of query variants to generate.
    """

    def __init__(self, num_queries: int = 3) -> None:
        self.num_queries = num_queries

    def generate(self, query: str) -> list[str]:
        """
        Generate reformulated versions of *query*.

        Parameters
        ----------
        query : str
            Original query.

        Returns
        -------
        list[str]
            The original query followed by up to ``num_queries - 1``
            reformulations (the original always comes first).
        """
        variants: list[str] = [query]

        synonym_variant = self._synonym_expansion(query)
        if synonym_variant != query:
            variants.append(synonym_variant)

        rephrased = self._rephrase(query)
        if rephrased != query:
            variants.append(rephrased)

        perspective = self._perspective_shift(query)
        if perspective != query:
            variants.append(perspective)

        keyword_variant = self._keyword_focus(query)
        if keyword_variant != query:
            variants.append(keyword_variant)

        seen: set[str] = set()
        unique: list[str] = []
        for v in variants:
            normalised = v.lower().strip()
            if normalised not in seen:
                seen.add(normalised)
                unique.append(v)

        result = unique[: self.num_queries]
        logger.debug(
            "Generated %d query variants for: '%s'",
            len(result),
            query,
        )
        return result

    @staticmethod
    def _synonym_expansion(query: str) -> str:
        """Replace one word with a synonym when a mapping exists."""
        words = query.split()
        for i, word in enumerate(words):
            key = word.lower().rstrip(".,?!;:")
            if key in _SYNONYM_MAP:
                synonyms = _SYNONYM_MAP[key]
                replacement = synonyms[0]
                if word[0].isupper():
                    replacement = replacement.capitalize()
                trailing = word[len(key) :]
                words[i] = replacement + trailing
                return " ".join(words)
        return query

    @staticmethod
    def _rephrase(query: str) -> str:
        """Structural rephrasing of the query."""
        q = query.strip().rstrip("?")

        patterns: list[tuple[re.Pattern[str], str]] = [
            (
                re.compile(r"^what is (.+)", re.I),
                r"explain \1",
            ),
            (
                re.compile(r"^how (?:do|does|can) (.+?) (.+)", re.I),
                r"what is the process for \1 to \2",
            ),
            (
                re.compile(r"^why (?:do|does|is|are) (.+)", re.I),
                r"what is the reason that \1",
            ),
            (
                re.compile(r"^explain (.+)", re.I),
                r"what is \1",
            ),
        ]
        for pattern, replacement in patterns:
            match = pattern.match(q)
            if match:
                rephrased = pattern.sub(replacement, q)
                return rephrased.strip() + "?"
        return query

    @staticmethod
    def _perspective_shift(query: str) -> str:
        """Reframe the query from a different angle."""
        q = query.strip().rstrip("?")

        if re.match(r"^(what|how|why|when|where|who)\b", q, re.I):
            return f"Describe the key aspects of {q.split(maxsplit=1)[-1]}?"

        return f"What should someone know about {q}?"

    @staticmethod
    def _keyword_focus(query: str) -> str:
        """Extract and foreground the most content-rich keywords."""
        stop = {
            "a",
            "an",
            "the",
            "is",
            "it",
            "of",
            "in",
            "to",
            "and",
            "or",
            "for",
            "on",
            "with",
            "as",
            "at",
            "by",
            "from",
            "that",
            "this",
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "which",
            "do",
            "does",
            "did",
            "can",
            "could",
            "would",
            "should",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
        }
        words = re.findall(r"\w+", query.lower())
        keywords = [w for w in words if w not in stop and len(w) > 2]
        if len(keywords) >= 2:
            return " ".join(keywords)
        return query


class QueryRouter:
    """
    Route queries to the most appropriate retriever.

    Analyses the query to determine its type (factual, conceptual,
    procedural, comparative, etc.) and dispatches to the retriever
    best suited for that type.

    Parameters
    ----------
    retrievers : dict[str, Any]
        Mapping of retriever names to retriever instances.  Each
        instance must expose a ``search(query, top_k)`` method.
    """

    _TYPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
        (
            "factual",
            re.compile(
                r"^(who|what|when|where)\b.*\??\s*$",
                re.I,
            ),
        ),
        (
            "procedural",
            re.compile(
                r"^how\b.*\??\s*$",
                re.I,
            ),
        ),
        (
            "conceptual",
            re.compile(
                r"\b(explain|describe|what is|define|concept)\b",
                re.I,
            ),
        ),
        (
            "comparative",
            re.compile(
                r"\b(compare|difference|versus|vs\.?|better|worse)\b",
                re.I,
            ),
        ),
        (
            "causal",
            re.compile(
                r"^why\b.*\??\s*$",
                re.I,
            ),
        ),
    ]

    def __init__(self, retrievers: dict[str, Any]) -> None:
        if not retrievers:
            raise ValueError("At least one retriever is required")
        self.retrievers = dict(retrievers)
        self._type_to_retriever: dict[str, str] = {}
        self._default_retriever = next(iter(self.retrievers))

    def set_routing(
        self,
        type_mapping: dict[str, str],
        default: str | None = None,
    ) -> None:
        """
        Configure which retriever handles which query type.

        Parameters
        ----------
        type_mapping : dict[str, str]
            Maps query types (``"factual"``, ``"procedural"``, etc.)
            to retriever names from ``self.retrievers``.
        default : str, optional
            Fallback retriever name.  Defaults to the first registered.
        """
        for qtype, rname in type_mapping.items():
            if rname not in self.retrievers:
                raise ValueError(
                    f"Retriever '{rname}' not found. Available: {list(self.retrievers.keys())}"
                )
            self._type_to_retriever[qtype] = rname

        if default is not None:
            if default not in self.retrievers:
                raise ValueError(f"Default retriever '{default}' not found")
            self._default_retriever = default

    def route(self, query: str) -> str:
        """
        Determine the best retriever name for *query*.

        Parameters
        ----------
        query : str
            The user's query.

        Returns
        -------
        str
            Name of the chosen retriever.
        """
        query_type = self._classify_query(query)

        retriever_name = self._type_to_retriever.get(
            query_type,
            self._default_retriever,
        )
        logger.debug(
            "Routed query (type=%s) → retriever '%s'",
            query_type,
            retriever_name,
        )
        return retriever_name

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Route the query and search with the selected retriever.

        Parameters
        ----------
        query : str
            Natural-language query.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[SearchResult]
            Results from the chosen retriever.
        """
        retriever_name = self.route(query)
        retriever = self.retrievers[retriever_name]
        logger.info(
            "QueryRouter: searching with '%s' (top_k=%d)",
            retriever_name,
            top_k,
        )
        return retriever.search(query, top_k=top_k)

    @classmethod
    def _classify_query(cls, query: str) -> str:
        """Classify *query* into a query-type string."""
        for qtype, pattern in cls._TYPE_PATTERNS:
            if pattern.search(query):
                return qtype
        return "general"
