"""
Callback system for training and pipeline events.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TrainingCallback:
    """Base class for training callbacks."""

    def on_train_start(self, info: dict[str, Any]) -> None:
        pass

    def on_train_end(self, info: dict[str, Any]) -> None:
        pass

    def on_epoch_end(self, info: dict[str, Any]) -> None:
        pass

    def on_step(self, info: dict[str, Any]) -> None:
        pass


class LoggingCallback(TrainingCallback):
    """Callback that logs training events."""

    def on_train_start(self, info: dict[str, Any]) -> None:
        logger.info(f"Training started: {info}")

    def on_train_end(self, info: dict[str, Any]) -> None:
        logger.info(f"Training finished: {info}")

    def on_epoch_end(self, info: dict[str, Any]) -> None:
        logger.info(f"Epoch complete: {info}")


class EarlyStoppingCallback(TrainingCallback):
    """Stop training when loss stops improving."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss: float | None = None
        self._counter = 0
        self.should_stop = False

    def on_epoch_end(self, info: dict[str, Any]) -> None:
        loss = info.get("avg_loss")
        if loss is None:
            return

        if self._best_loss is None or loss < self._best_loss - self.min_delta:
            self._best_loss = loss
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered: no improvement for {self.patience} epochs"
                )


class CallbackManager:
    """Manages a list of callbacks."""

    def __init__(self) -> None:
        self._callbacks: list[TrainingCallback] = []

    def add(self, callback: TrainingCallback) -> None:
        self._callbacks.append(callback)

    def on_train_start(self, info: dict[str, Any]) -> None:
        for cb in self._callbacks:
            cb.on_train_start(info)

    def on_train_end(self, info: dict[str, Any]) -> None:
        for cb in self._callbacks:
            cb.on_train_end(info)

    def on_epoch_end(self, info: dict[str, Any]) -> None:
        for cb in self._callbacks:
            cb.on_epoch_end(info)

    def on_step(self, info: dict[str, Any]) -> None:
        for cb in self._callbacks:
            cb.on_step(info)
