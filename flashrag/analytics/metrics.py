"""
Retrieval and generation evaluation metrics.

Implements standard IR metrics: Recall@K, MRR, NDCG, as well as
RAG-specific metrics: faithfulness and relevance scoring.
"""

from __future__ import annotations

import math

import numpy as np


def compute_recall_at_k(
    retrieved: list[list[str]],
    relevant: list[list[str]],
    k: int,
) -> float:
    """
    Compute Recall@K averaged across queries.

    Recall@K = |retrieved_top_k ∩ relevant| / |relevant|

    Parameters
    ----------
    retrieved : list of list of str
        Retrieved document texts per query.
    relevant : list of list of str
        Ground-truth relevant document texts per query.
    k : int
        Number of top results to consider.
    """
    if not retrieved:
        return 0.0

    recalls = []
    for ret, rel in zip(retrieved, relevant):
        if not rel:
            continue
        top_k = set(ret[:k])
        rel_set = set(rel)
        hit = len(top_k & rel_set)
        recalls.append(hit / len(rel_set))

    return float(np.mean(recalls)) if recalls else 0.0


def compute_mrr(
    retrieved: list[list[str]],
    relevant: list[list[str]],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    MRR = mean over queries of 1/rank of first relevant result.
    """
    if not retrieved:
        return 0.0

    reciprocal_ranks = []
    for ret, rel in zip(retrieved, relevant):
        rel_set = set(rel)
        rr = 0.0
        for rank, doc in enumerate(ret, start=1):
            if doc in rel_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return float(np.mean(reciprocal_ranks))


def compute_ndcg(
    retrieved: list[list[str]],
    relevant: list[list[str]],
    k: int,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K).

    Uses binary relevance: 1 if document is in the relevant set, 0 otherwise.
    """
    if not retrieved:
        return 0.0

    ndcgs = []
    for ret, rel in zip(retrieved, relevant):
        rel_set = set(rel)
        dcg = 0.0
        for i, doc in enumerate(ret[:k]):
            if doc in rel_set:
                dcg += 1.0 / math.log2(i + 2)

        ideal_hits = min(len(rel_set), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs))


def compute_faithfulness(
    answer: str,
    contexts: list[str],
    tokenize_fn: callable | None = None,
) -> float:
    """
    Compute faithfulness score: how much of the answer is grounded in the contexts.

    Uses token-level overlap as a proxy. A faithfulness of 1.0 means every
    word in the answer appears in at least one context passage.

    Parameters
    ----------
    answer : str
        Generated answer text.
    contexts : list of str
        Context passages used for generation.
    tokenize_fn : callable, optional
        Custom tokenizer. Defaults to whitespace + lowercasing.
    """
    if not answer or not contexts:
        return 0.0

    tokenize = tokenize_fn or (lambda s: set(s.lower().split()))

    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 0.0

    context_tokens: set[str] = set()
    for ctx in contexts:
        context_tokens |= tokenize(ctx)

    grounded = answer_tokens & context_tokens
    return len(grounded) / len(answer_tokens)


def compute_relevance(
    answer: str,
    question: str,
    tokenize_fn: callable | None = None,
) -> float:
    """
    Compute answer relevance: how well the answer addresses the question.

    Uses token-level overlap between question and answer as a simple
    proxy for relevance.
    """
    if not answer or not question:
        return 0.0

    tokenize = tokenize_fn or (lambda s: set(s.lower().split()))

    question_tokens = tokenize(question)
    answer_tokens = tokenize(answer)

    stop_words = {
        "a", "an", "the", "is", "it", "of", "in", "to", "and", "or",
        "for", "on", "with", "as", "at", "by", "what", "how", "why",
        "when", "where", "who", "which", "do", "does", "did", "?",
    }
    question_content = question_tokens - stop_words
    answer_content = answer_tokens - stop_words

    if not question_content:
        return 0.0

    overlap = question_content & answer_content
    return len(overlap) / len(question_content)


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())

    if not pred_tokens or not truth_tokens:
        return 0.0

    common = pred_tokens & truth_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Check if the prediction exactly matches the ground truth (normalized)."""
    pred_norm = " ".join(prediction.lower().split())
    truth_norm = " ".join(ground_truth.lower().split())
    return 1.0 if pred_norm == truth_norm else 0.0
