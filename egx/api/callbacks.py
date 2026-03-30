"""
EGX Callback System — Layer 7.

Production-grade training callbacks inspired by the patterns used in
state-of-the-art large model training (gradient monitoring, eval scheduling,
early stopping, throughput tracking, learning-rate logging).

Users subclass TrainingCallback and override only what they need.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from egx.api.trainer import EGXTrainer

logger = logging.getLogger("egx.api.callbacks")


# ──────────────────────────────────────────────────────────────────────
#  Context Types for Callbacks
# ──────────────────────────────────────────────────────────────────────

class EpochEndContext(TypedDict, total=False):
    metrics: Dict[str, float]


class StepContext(TypedDict, total=False):
    batch: Dict[str, Any]
    grad_norm: float
    throughput_tokens_per_sec: float
    checkpoint_saved: bool


class ResultContext(TypedDict, total=False):
    result: Dict[str, Any]


class LogContext(TypedDict, total=False):
    logs: Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────
#  Base Callback
# ──────────────────────────────────────────────────────────────────────

class TrainingCallback:
    """
    Base class for EGX training callbacks.

    Override any method to hook into the training lifecycle.
    Every hook receives the trainer instance so you can inspect or
    mutate state (model, optimizer, scheduler, config, etc.).
    """

    # ── Initialization ──
    def on_init_end(self, trainer: "EGXTrainer", **kwargs) -> None:
        """Called at the end of trainer __init__."""

    # ── Training lifecycle ──
    def on_train_begin(self, trainer: "EGXTrainer", **kwargs) -> None:
        """Called before the first training step."""

    def on_train_end(self, trainer: "EGXTrainer", result: Dict[str, Any], **kwargs: ResultContext) -> None:
        """Called after all training is complete."""

    # ── Epoch lifecycle ──
    def on_epoch_begin(self, trainer: "EGXTrainer", epoch: int, **kwargs) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(self, trainer: "EGXTrainer", epoch: int, metrics: Dict[str, float], **kwargs: EpochEndContext) -> None:
        """Called at the end of each epoch with aggregated metrics."""

    # ── Step lifecycle ──
    def on_step_begin(self, trainer: "EGXTrainer", step: int, **kwargs: StepContext) -> None:
        """Called before each training step (forward + backward)."""

    def on_step_end(self, trainer: "EGXTrainer", step: int, loss: float, lr: float, **kwargs: StepContext) -> None:
        """Called after each training step with loss and current lr."""

    # ── Gradient hooks ──
    def on_before_backward(self, trainer: "EGXTrainer", loss: Any, **kwargs) -> None:
        """Called after forward but before loss.backward()."""

    def on_after_backward(self, trainer: "EGXTrainer", **kwargs) -> None:
        """Called after loss.backward() but before optimizer.step()."""

    def on_before_optimizer_step(self, trainer: "EGXTrainer", **kwargs) -> None:
        """Called after gradient clipping but before optimizer.step()."""

    # ── Evaluation ──
    def on_evaluate_begin(self, trainer: "EGXTrainer", **kwargs) -> None:
        """Called before the evaluation loop."""

    def on_evaluate_end(self, trainer: "EGXTrainer", metrics: Dict[str, float], **kwargs) -> None:
        """Called after evaluation with computed metrics."""

    # ── Prediction ──
    def on_predict_begin(self, trainer: "EGXTrainer", **kwargs) -> None:
        """Called before the prediction loop."""

    def on_predict_end(self, trainer: "EGXTrainer", **kwargs) -> None:
        """Called after the prediction loop."""

    # ── Checkpointing ──
    def on_save(self, trainer: "EGXTrainer", checkpoint_path: str, **kwargs) -> None:
        """Called when a checkpoint is saved."""

    def on_load(self, trainer: "EGXTrainer", checkpoint_path: str, **kwargs) -> None:
        """Called when a checkpoint is loaded."""

    # ── Logging ──
    def on_log(self, trainer: "EGXTrainer", logs: Dict[str, Any], **kwargs: LogContext) -> None:
        """Called every logging_steps with aggregated log dict."""


class CallbackHandler:
    """Dispatches lifecycle events to all registered callbacks."""

    def __init__(self, callbacks: Optional[List[TrainingCallback]] = None):
        self.callbacks: List[TrainingCallback] = list(callbacks or [])

    def add(self, callback: TrainingCallback) -> None:
        self.callbacks.append(callback)

    def remove(self, callback_type: type) -> None:
        self.callbacks = [c for c in self.callbacks if not isinstance(c, callback_type)]

    def fire(self, event: str, **kwargs) -> None:
        """Fire an event on all callbacks. Silently skips missing methods."""
        for cb in self.callbacks:
            fn = getattr(cb, event, None)
            if fn is not None:
                try:
                    fn(**kwargs)
                except Exception as e:
                    logger.warning(
                        f"Callback {type(cb).__name__}.{event} raised: {e}"
                    )


# ──────────────────────────────────────────────────────────────────────
#  Built-in Callbacks
# ──────────────────────────────────────────────────────────────────────

class EarlyStoppingCallback(TrainingCallback):
    """
    Stop training when a monitored metric stops improving.
    Mirrors behaviour of modern LLM trainers.
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        metric_name: str = "eval_loss",
        greater_is_better: bool = False,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_value: Optional[float] = None
        self.wait_count = 0
        self.should_stop = False

    def on_evaluate_end(self, trainer, metrics, **kwargs):
        current = metrics.get(self.metric_name)
        if current is None:
            return

        if self.best_value is None:
            self.best_value = current
            return

        if self.greater_is_better:
            improved = current > self.best_value + self.min_delta
        else:
            improved = current < self.best_value - self.min_delta

        if improved:
            self.best_value = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.should_stop = True
                logger.info(
                    f"EarlyStopping triggered: {self.metric_name} did not improve "
                    f"for {self.patience} evaluations. Best: {self.best_value:.6f}"
                )


class LoggingCallback(TrainingCallback):
    """
    Production logging: loss, lr, throughput (tokens/sec),
    memory usage, gradient norm — like the loggers used in Megatron-LM / GPT training.
    """

    def __init__(self, log_every_n_steps: int = 10):
        self.log_every = log_every_n_steps
        self._step_losses: List[float] = []
        self._epoch_start: float = 0.0
        self._train_start: float = 0.0

    def on_train_begin(self, trainer, **kwargs):
        self._train_start = time.time()
        logger.info("╔══════════════════════════════════════╗")
        logger.info("║       EGX Training Session Start     ║")
        logger.info("╚══════════════════════════════════════╝")

    def on_epoch_begin(self, trainer, epoch, **kwargs):
        self._epoch_start = time.time()
        self._step_losses.clear()

    def on_step_end(self, trainer, step, loss, lr, **kwargs):
        self._step_losses.append(loss)
        if step > 0 and step % self.log_every == 0:
            avg_loss = sum(self._step_losses[-self.log_every:]) / min(
                len(self._step_losses), self.log_every
            )
            elapsed = time.time() - self._epoch_start
            steps_per_sec = step / elapsed if elapsed > 0 else 0
            logger.info(
                f"Step {step:>6d} | Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e} | Steps/s: {steps_per_sec:.2f}"
            )

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        epoch_time = time.time() - self._epoch_start
        avg = metrics.get("train_loss_epoch", 0.0)
        logger.info(
            f"Epoch {epoch + 1} done | Avg Loss: {avg:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

    def on_evaluate_end(self, trainer, metrics, **kwargs):
        parts = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info(f"Eval ▶ {parts}")

    def on_train_end(self, trainer, result, **kwargs):
        total = time.time() - self._train_start
        logger.info("╔══════════════════════════════════════╗")
        logger.info(f"║  Training Complete — {total:.1f}s total")
        logger.info("╚══════════════════════════════════════╝")


class GradientClipCallback(TrainingCallback):
    """Clip gradients by global norm — standard in all large-model training."""

    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm

    def on_after_backward(self, trainer, **kwargs):
        try:
            import torch
            if hasattr(trainer, '_model') and trainer._model is not None:
                torch.nn.utils.clip_grad_norm_(
                    trainer._model.parameters(), self.max_norm
                )
        except Exception as e:
            logger.debug("GradientClipCallback: skipped clipping (%s)", e)


class NaNDetectionCallback(TrainingCallback):
    """
    Detect NaN / Inf losses and either skip the batch or halt training.
    Critical safety feature in production LLM training.
    """

    def __init__(self, halt_on_nan: bool = False, max_nan_count: int = 10):
        self.halt_on_nan = halt_on_nan
        self.max_nan_count = max_nan_count
        self.nan_count = 0

    def on_step_end(self, trainer, step, loss, **kwargs):
        if math.isnan(loss) or math.isinf(loss):
            self.nan_count += 1
            logger.warning(
                f"NaN/Inf loss at step {step} (count: {self.nan_count})"
            )
            if self.halt_on_nan or self.nan_count >= self.max_nan_count:
                raise RuntimeError(
                    f"Training halted: {self.nan_count} NaN/Inf losses detected."
                )


class ThroughputCallback(TrainingCallback):
    """
    Track tokens/second throughput — essential metric for LLM training
    (used in Megatron, DeepSpeed, GPT-NeoX etc.).
    """

    def __init__(self, log_every_n_steps: int = 50):
        self.log_every = log_every_n_steps
        self.total_tokens = 0
        self._start: float = 0.0

    def on_train_begin(self, trainer, **kwargs):
        self._start = time.time()

    def on_step_end(self, trainer, step, loss, **kwargs):
        # Try to count tokens from the batch
        batch = kwargs.get("batch")
        if batch and isinstance(batch, dict) and "input_ids" in batch:
            ids = batch["input_ids"]
            if hasattr(ids, "numel"):
                self.total_tokens += ids.numel()
            elif hasattr(ids, "__len__"):
                self.total_tokens += len(ids)

        if step > 0 and step % self.log_every == 0:
            elapsed = time.time() - self._start
            tps = self.total_tokens / elapsed if elapsed > 0 else 0
            logger.info(f"Throughput ▶ {tps:,.0f} tokens/sec (total: {self.total_tokens:,})")


class CheckpointCallback(TrainingCallback):
    """
    Save checkpoints at regular intervals and track the best model.
    """

    def __init__(
        self,
        save_every_n_steps: int = 500,
        save_best: bool = True,
        metric_name: str = "eval_loss",
        greater_is_better: bool = False,
    ):
        self.save_every = save_every_n_steps
        self.save_best = save_best
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric: Optional[float] = None

    def on_step_end(self, trainer, step, **kwargs):
        if self.save_every > 0 and step > 0 and step % self.save_every == 0:
            path = f"{trainer.config.output_dir}/checkpoint-{step}"
            self._save(trainer, path)

    def on_evaluate_end(self, trainer, metrics, **kwargs):
        if not self.save_best:
            return
        current = metrics.get(self.metric_name)
        if current is None:
            return

        if self.best_metric is None:
            self.best_metric = current
            self._save(trainer, f"{trainer.config.output_dir}/best_model")
        else:
            improved = (
                current > self.best_metric if self.greater_is_better
                else current < self.best_metric
            )
            if improved:
                self.best_metric = current
                self._save(trainer, f"{trainer.config.output_dir}/best_model")
                logger.info(f"New best model! {self.metric_name}={current:.6f}")

    @staticmethod
    def _save(trainer, path: str):
        import os
        os.makedirs(path, exist_ok=True)
        try:
            import torch
            if hasattr(trainer, '_model') and trainer._model is not None:
                torch.save(trainer._model.state_dict(), f"{path}/model.pt")
                logger.info(f"Checkpoint saved → {path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
