"""
Benchmarking suite for RAG systems.

Measures retrieval quality, generation quality, and end-to-end
pipeline performance across standard metrics.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from flashrag.analytics.metrics import (
    compute_exact_match,
    compute_f1,
    compute_faithfulness,
    compute_mrr,
    compute_ndcg,
    compute_recall_at_k,
    compute_relevance,
)
from flashrag.pipelines.basic_rag import BasicRAGPipeline, RAGResult

logger = logging.getLogger(__name__)


class Benchmark:
    """
    Comprehensive RAG benchmarking suite.

    Parameters
    ----------
    pipeline : BasicRAGPipeline, optional
        Pipeline to benchmark. Can also be set later.
    output_dir : str
        Directory for benchmark results.
    """

    def __init__(
        self,
        pipeline: BasicRAGPipeline | None = None,
        output_dir: str = "workspace/benchmark",
    ) -> None:
        self._pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_pipeline(self, pipeline: BasicRAGPipeline) -> None:
        self._pipeline = pipeline

    def run(
        self,
        eval_data: list[dict[str, Any]] | None = None,
        eval_path: str | Path | None = None,
        ks: list[int] | None = None,
        save_results: bool = True,
    ) -> dict[str, Any]:
        """
        Run a full benchmark evaluation.

        Each item in eval_data should have:
        - ``question`` (str): the query
        - ``answer`` (str, optional): ground-truth answer
        - ``relevant_docs`` (list[str], optional): ground-truth relevant texts

        Parameters
        ----------
        eval_data : list of dict
            Evaluation dataset.
        eval_path : str, optional
            Path to JSONL evaluation file.
        ks : list of int
            K values for retrieval metrics.
        save_results : bool
            Save results to JSON.
        """
        if not self._pipeline:
            raise RuntimeError("No pipeline set. Use set_pipeline() first.")

        if eval_path and not eval_data:
            eval_data = []
            with open(eval_path) as f:
                for line in f:
                    eval_data.append(json.loads(line))

        if not eval_data:
            raise ValueError("Provide eval_data or eval_path")

        ks = ks or [1, 3, 5, 10]

        questions = [d["question"] for d in eval_data]
        gt_answers = [d.get("answer") for d in eval_data]
        gt_relevant = [d.get("relevant_docs", []) for d in eval_data]

        logger.info(f"Running benchmark on {len(questions)} queries...")
        start_time = time.time()

        results: list[RAGResult] = []
        latencies: list[float] = []

        for question in questions:
            t0 = time.time()
            result = self._pipeline.run(question)
            latencies.append(time.time() - t0)
            results.append(result)

        total_time = time.time() - start_time

        metrics: dict[str, Any] = {"num_queries": len(questions)}

        retrieved_texts = [r.contexts for r in results]
        has_relevant = any(docs for docs in gt_relevant)
        if has_relevant:
            for k in ks:
                metrics[f"recall@{k}"] = compute_recall_at_k(retrieved_texts, gt_relevant, k)
                metrics[f"ndcg@{k}"] = compute_ndcg(retrieved_texts, gt_relevant, k)
            metrics["mrr"] = compute_mrr(retrieved_texts, gt_relevant)

        has_answers = any(a for a in gt_answers)
        if has_answers:
            f1_scores = []
            em_scores = []
            faith_scores = []
            rel_scores = []

            for result, gt_ans, question in zip(results, gt_answers, questions):
                if gt_ans:
                    f1_scores.append(compute_f1(result.answer, gt_ans))
                    em_scores.append(compute_exact_match(result.answer, gt_ans))

                if result.contexts:
                    faith_scores.append(
                        compute_faithfulness(result.answer, result.contexts)
                    )
                rel_scores.append(compute_relevance(result.answer, question))

            if f1_scores:
                metrics["f1"] = float(np.mean(f1_scores))
                metrics["exact_match"] = float(np.mean(em_scores))
            if faith_scores:
                metrics["faithfulness"] = float(np.mean(faith_scores))
            if rel_scores:
                metrics["relevance"] = float(np.mean(rel_scores))

        metrics["avg_latency_s"] = float(np.mean(latencies))
        metrics["p95_latency_s"] = float(np.percentile(latencies, 95))
        metrics["total_time_s"] = total_time
        metrics["queries_per_second"] = len(questions) / total_time if total_time > 0 else 0

        if save_results:
            results_path = self.output_dir / "benchmark_results.json"
            with open(results_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Benchmark results saved to {results_path}")

        logger.info(f"Benchmark complete: {metrics}")
        return metrics
