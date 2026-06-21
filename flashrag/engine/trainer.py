"""
RAG system trainer.

Handles fine-tuning of embedding models and generator models on
domain-specific retrieval and QA datasets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from flashrag.cfg.config import RAGConfig, get_config
from flashrag.utils.callbacks import CallbackManager, TrainingCallback

logger = logging.getLogger(__name__)


class _QADataset(Dataset):
    """Simple Q&A dataset for RAG training."""

    def __init__(self, data: list[dict[str, str]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.data[idx]


class RAGTrainer:
    """
    Trainer for RAG system components.

    Supports fine-tuning embedding models on contrastive pairs and
    generator models on question-answer datasets with retrieved context.

    Parameters
    ----------
    config : RAGConfig, optional
        Full configuration object.
    output_dir : str
        Directory for saving checkpoints and logs.
    learning_rate : float
        Optimizer learning rate.
    epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    device : str
        Device for training.
    """

    def __init__(
        self,
        config: RAGConfig | None = None,
        output_dir: str = "workspace/train",
        learning_rate: float = 2e-5,
        epochs: int = 3,
        batch_size: int = 8,
        device: str = "cpu",
    ) -> None:
        self.config = config or get_config()
        self.output_dir = Path(output_dir)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.callback_manager = CallbackManager()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_callback(self, callback: TrainingCallback) -> None:
        self.callback_manager.add(callback)

    def train_embedding(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        train_pairs: list[tuple[str, str]] | None = None,
        train_path: str | None = None,
    ) -> dict[str, float]:
        """
        Fine-tune a sentence-transformer embedding model on contrastive pairs.

        Parameters
        ----------
        model_name : str
            Base embedding model to fine-tune.
        train_pairs : list of (query, positive_doc) tuples
            Training pairs for contrastive learning.
        train_path : str, optional
            Path to JSONL file with ``{"query": ..., "positive": ...}`` entries.
        """
        from sentence_transformers import InputExample, SentenceTransformer, losses

        if train_path and not train_pairs:
            train_pairs = []
            with open(train_path) as f:
                for line in f:
                    item = json.loads(line)
                    train_pairs.append((item["query"], item["positive"]))

        if not train_pairs:
            raise ValueError("Provide train_pairs or train_path")

        model = SentenceTransformer(model_name, device=self.device)
        examples = [InputExample(texts=[q, p]) for q, p in train_pairs]
        dataloader = DataLoader(examples, shuffle=True, batch_size=self.batch_size)
        loss = losses.MultipleNegativesRankingLoss(model)

        self.callback_manager.on_train_start({"model": model_name, "pairs": len(train_pairs)})

        save_path = str(self.output_dir / "embedding_finetuned")
        model.fit(
            train_objectives=[(dataloader, loss)],
            epochs=self.epochs,
            output_path=save_path,
            show_progress_bar=True,
        )

        self.callback_manager.on_train_end({"save_path": save_path})
        logger.info(f"Embedding model fine-tuned and saved to {save_path}")

        return {"save_path": save_path, "train_pairs": len(train_pairs)}

    def train_generator(
        self,
        model_name: str = "gpt2",
        train_data: list[dict[str, str]] | None = None,
        train_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Fine-tune a generator model on RAG Q&A data.

        Each training example should have ``context``, ``question``, and ``answer``.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if train_path and not train_data:
            train_data = []
            with open(train_path) as f:
                for line in f:
                    train_data.append(json.loads(line))

        if not train_data:
            raise ValueError("Provide train_data or train_path")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        model.train()

        self.callback_manager.on_train_start(
            {"model": model_name, "examples": len(train_data)}
        )

        total_loss = 0.0
        steps = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            pbar = tqdm(train_data, desc=f"Epoch {epoch + 1}/{self.epochs}")

            for item in pbar:
                text = (
                    f"Context: {item.get('context', '')}\n"
                    f"Question: {item['question']}\n"
                    f"Answer: {item['answer']}"
                )
                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True,
                    max_length=512, padding="max_length"
                ).to(self.device)

                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                steps += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

                self.callback_manager.on_step(
                    {"step": steps, "loss": loss.item(), "epoch": epoch + 1}
                )

            avg_loss = epoch_loss / len(train_data)
            total_loss += avg_loss
            self.callback_manager.on_epoch_end(
                {"epoch": epoch + 1, "avg_loss": avg_loss}
            )
            logger.info(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

        save_path = self.output_dir / "generator_finetuned"
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))

        self.callback_manager.on_train_end({"save_path": str(save_path)})
        logger.info(f"Generator saved to {save_path}")

        return {
            "save_path": str(save_path),
            "final_loss": total_loss / self.epochs,
            "total_steps": steps,
        }
