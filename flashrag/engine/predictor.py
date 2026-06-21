"""
RAG predictor for batch inference.

Runs the full RAG pipeline on a list of questions and collects results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from flashrag.pipelines.basic_rag import BasicRAGPipeline, RAGResult

logger = logging.getLogger(__name__)


class RAGPredictor:
    """
    Batch prediction engine for RAG pipelines.

    Parameters
    ----------
    pipeline : BasicRAGPipeline
        A configured RAG pipeline.
    """

    def __init__(self, pipeline: BasicRAGPipeline) -> None:
        self._pipeline = pipeline

    def predict(
        self,
        questions: List[str],
        show_progress: bool = True,
        **kwargs: Any,
    ) -> List[RAGResult]:
        """Run the pipeline on a list of questions."""
        results: List[RAGResult] = []
        iterator = tqdm(questions, desc="Predicting") if show_progress else questions

        for question in iterator:
            result = self._pipeline.run(question, **kwargs)
            results.append(result)

        logger.info(f"Predicted {len(results)} questions")
        return results

    def predict_from_file(
        self,
        input_path: str | Path,
        output_path: Optional[str | Path] = None,
        question_key: str = "question",
        **kwargs: Any,
    ) -> List[RAGResult]:
        """
        Read questions from a JSONL file and write results.
        """
        questions: List[str] = []
        with open(input_path) as f:
            for line in f:
                item = json.loads(line)
                questions.append(item[question_key])

        results = self.predict(questions, **kwargs)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for q, r in zip(questions, results):
                    json.dump(
                        {
                            "question": q,
                            "answer": r.answer,
                            "num_sources": len(r.sources),
                        },
                        f,
                    )
                    f.write("\n")
            logger.info(f"Results saved to {output_path}")

        return results
