"""
RAGAS-style evaluation metrics for Retrieval-Augmented Generation.

Implements context precision, context recall, answer faithfulness, and
answer relevance as standalone functions plus a convenience evaluator class
that computes all metrics in one call.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset(
    {
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
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "that",
        "this",
        "these",
        "those",
        "are",
        "but",
        "if",
        "not",
        "no",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
        "should",
        "now",
        "its",
        "from",
        "also",
        "more",
        "some",
        "any",
        "?",
        ".",
        ",",
        "!",
        ";",
        ":",
        "'",
        '"',
    }
)


def _default_tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenizer with basic punctuation stripping."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens


def _content_tokens(text: str, tokenize_fn: Callable | None = None) -> set[str]:
    """Tokenize and remove stop words."""
    tokenize = tokenize_fn or _default_tokenize
    tokens = tokenize(text) if callable(tokenize) else _default_tokenize(text)
    return set(tokens) - _STOP_WORDS


# ---------------------------------------------------------------------------
# Standalone metric functions
# ---------------------------------------------------------------------------


def compute_context_precision(
    questions: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    tokenize_fn: Callable[[str], list[str]] | None = None,
) -> float:
    """
    Compute Context Precision averaged across queries.

    Measures how many of the retrieved contexts are relevant to the
    ground-truth answer by computing token overlap between each context
    and the ground truth, then averaging the precision per query.

    Parameters
    ----------
    questions : list of str
        Input questions (used for alignment; not scored directly).
    contexts : list of list of str
        Retrieved context passages per query.
    ground_truths : list of str
        Ground-truth answers per query.
    tokenize_fn : callable or None
        Custom tokenizer returning a list of token strings.

    Returns
    -------
    float
        Mean context precision in [0, 1].
    """
    if not questions or not contexts or not ground_truths:
        return 0.0

    precisions: list[float] = []
    for ctxs, gt in zip(contexts, ground_truths):
        if not ctxs:
            precisions.append(0.0)
            continue

        gt_tokens = _content_tokens(gt, tokenize_fn)
        if not gt_tokens:
            precisions.append(0.0)
            continue

        relevant_count = 0
        for ctx in ctxs:
            ctx_tokens = _content_tokens(ctx, tokenize_fn)
            overlap = ctx_tokens & gt_tokens
            if len(overlap) / max(len(gt_tokens), 1) > 0.1:
                relevant_count += 1

        precisions.append(relevant_count / len(ctxs))

    return float(np.mean(precisions)) if precisions else 0.0


def compute_context_recall(
    questions: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    tokenize_fn: Callable[[str], list[str]] | None = None,
) -> float:
    """
    Compute Context Recall averaged across queries.

    Measures how much of the ground truth is covered by the retrieved
    contexts. Calculated as the fraction of ground-truth content tokens
    that appear in at least one retrieved context.

    Parameters
    ----------
    questions : list of str
        Input questions.
    contexts : list of list of str
        Retrieved context passages per query.
    ground_truths : list of str
        Ground-truth answers per query.
    tokenize_fn : callable or None
        Custom tokenizer returning a list of token strings.

    Returns
    -------
    float
        Mean context recall in [0, 1].
    """
    if not questions or not contexts or not ground_truths:
        return 0.0

    recalls: list[float] = []
    for ctxs, gt in zip(contexts, ground_truths):
        gt_tokens = _content_tokens(gt, tokenize_fn)
        if not gt_tokens:
            recalls.append(0.0)
            continue

        all_ctx_tokens: set[str] = set()
        for ctx in ctxs:
            all_ctx_tokens |= _content_tokens(ctx, tokenize_fn)

        covered = gt_tokens & all_ctx_tokens
        recalls.append(len(covered) / len(gt_tokens))

    return float(np.mean(recalls)) if recalls else 0.0


def compute_answer_faithfulness(
    answers: list[str],
    contexts: list[list[str]],
    tokenize_fn: Callable[[str], list[str]] | None = None,
) -> float:
    """
    Compute Answer Faithfulness averaged across samples.

    Measures how well the generated answer is grounded in its retrieved
    contexts by decomposing the answer into sentences and checking token
    overlap of each sentence against the combined contexts.

    Parameters
    ----------
    answers : list of str
        Generated answers.
    contexts : list of list of str
        Context passages used for each answer.
    tokenize_fn : callable or None
        Custom tokenizer returning a list of token strings.

    Returns
    -------
    float
        Mean faithfulness score in [0, 1].
    """
    if not answers or not contexts:
        return 0.0

    sentence_re = re.compile(r"(?<=[.!?])\s+")
    scores: list[float] = []

    for answer, ctxs in zip(answers, contexts):
        if not answer.strip() or not ctxs:
            scores.append(0.0)
            continue

        ctx_tokens: set[str] = set()
        for ctx in ctxs:
            ctx_tokens |= _content_tokens(ctx, tokenize_fn)

        if not ctx_tokens:
            scores.append(0.0)
            continue

        claims = sentence_re.split(answer.strip())
        claims = [c.strip() for c in claims if c.strip()]
        if not claims:
            scores.append(0.0)
            continue

        grounded_count = 0
        for claim in claims:
            claim_tokens = _content_tokens(claim, tokenize_fn)
            if not claim_tokens:
                grounded_count += 1
                continue
            overlap = claim_tokens & ctx_tokens
            if len(overlap) / len(claim_tokens) >= 0.5:
                grounded_count += 1

        scores.append(grounded_count / len(claims))

    return float(np.mean(scores)) if scores else 0.0


def compute_answer_relevance(
    questions: list[str],
    answers: list[str],
    tokenize_fn: Callable[[str], list[str]] | None = None,
) -> float:
    """
    Compute Answer Relevance averaged across samples.

    Measures how relevant the generated answer is to the input question
    using bidirectional token overlap (F1) between question and answer
    content tokens (stop words removed).

    Parameters
    ----------
    questions : list of str
        Input questions.
    answers : list of str
        Generated answers.
    tokenize_fn : callable or None
        Custom tokenizer returning a list of token strings.

    Returns
    -------
    float
        Mean answer relevance in [0, 1].
    """
    if not questions or not answers:
        return 0.0

    scores: list[float] = []
    for question, answer in zip(questions, answers):
        q_tokens = _content_tokens(question, tokenize_fn)
        a_tokens = _content_tokens(answer, tokenize_fn)

        if not q_tokens or not a_tokens:
            scores.append(0.0)
            continue

        overlap = q_tokens & a_tokens
        precision = len(overlap) / len(a_tokens)
        recall = len(overlap) / len(q_tokens)

        if precision + recall == 0:
            scores.append(0.0)
        else:
            f1 = 2 * precision * recall / (precision + recall)
            scores.append(f1)

    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------


class RAGASEvaluator:
    """
    Convenience class that computes multiple RAGAS metrics in one call.

    Parameters
    ----------
    metrics : list of str or None
        Subset of metrics to compute. Valid names:
        ``"context_precision"``, ``"context_recall"``,
        ``"answer_faithfulness"``, ``"answer_relevance"``.
        Defaults to all four.
    """

    _METRIC_REGISTRY: dict[str, Callable] = {
        "context_precision": compute_context_precision,
        "context_recall": compute_context_recall,
        "answer_faithfulness": compute_answer_faithfulness,
        "answer_relevance": compute_answer_relevance,
    }

    def __init__(self, metrics: list[str] | None = None) -> None:
        if metrics is None:
            self._metrics = list(self._METRIC_REGISTRY.keys())
        else:
            for m in metrics:
                if m not in self._METRIC_REGISTRY:
                    raise ValueError(
                        f"Unknown metric '{m}'. Available: {list(self._METRIC_REGISTRY.keys())}"
                    )
            self._metrics = list(metrics)

    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Evaluate all configured metrics over a batch of samples.

        Parameters
        ----------
        questions : list of str
            Input questions.
        answers : list of str
            Generated answers.
        contexts : list of list of str
            Retrieved context passages per query.
        ground_truths : list of str or None
            Ground-truth answers; required for context precision/recall.

        Returns
        -------
        dict[str, float]
            Mapping of metric name to its computed score.
        """
        results: dict[str, float] = {}

        for name in self._metrics:
            if name == "context_precision":
                if ground_truths is None:
                    logger.warning("Skipping context_precision: ground_truths not provided")
                    continue
                results[name] = compute_context_precision(questions, contexts, ground_truths)

            elif name == "context_recall":
                if ground_truths is None:
                    logger.warning("Skipping context_recall: ground_truths not provided")
                    continue
                results[name] = compute_context_recall(questions, contexts, ground_truths)

            elif name == "answer_faithfulness":
                results[name] = compute_answer_faithfulness(answers, contexts)

            elif name == "answer_relevance":
                results[name] = compute_answer_relevance(questions, answers)

        return results

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> dict[str, float]:
        """
        Evaluate metrics for a single query.

        Parameters
        ----------
        question : str
            Input question.
        answer : str
            Generated answer.
        contexts : list of str
            Retrieved context passages.
        ground_truth : str or None
            Ground-truth answer.

        Returns
        -------
        dict[str, float]
            Mapping of metric name to its computed score.
        """
        gt = [ground_truth] if ground_truth is not None else None
        return self.evaluate(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=gt,
        )
