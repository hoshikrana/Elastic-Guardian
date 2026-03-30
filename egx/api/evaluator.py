"""
EGX Evaluator — Layer 7.

Production-grade evaluation engine for model assessment.
Supports loss computation, perplexity, custom metrics, and
eval-during-training patterns used in modern LLM pipelines.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, Optional

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from egx.core.device import get_default_device
except ImportError:
    torch = None

logger = logging.getLogger("egx.api.evaluator")


class EGXEvaluator:
    """
    Standalone evaluator that can be used independently or embedded in EGXTrainer.

    Supports:
    - Loss computation over eval set
    - Perplexity calculation
    - Custom metric functions
    - Mixed-precision evaluation
    - Batched evaluation with configurable batch size
    """

    def __init__(
        self,
        metrics_fns: Optional[Dict[str, Callable]] = None,
        batch_size: int = 4,
        data_collator: Optional[Callable] = None,
    ):
        self.metrics_fns = metrics_fns or {}
        self.batch_size = batch_size
        self.data_collator = data_collator
        self.device = get_default_device()

    def __repr__(self) -> str:
        return f"EGXEvaluator(batch_size={self.batch_size}, device='{self.device}')"

    def evaluate(
        self,
        model: nn.Module,
        eval_dataset: Any,
        loss_fn: Optional[Callable] = None,
        compute_perplexity: bool = True,
        device: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Run a full evaluation pass over the dataset.

        Args:
            model: The model to evaluate.
            eval_dataset: Evaluation dataset (list of dicts or torch Dataset).
            loss_fn: Optional custom loss function. If None, uses model.loss.
            compute_perplexity: Whether to compute perplexity from loss.
            device: Device override.

        Returns:
            Dictionary with eval_loss, eval_perplexity, and any custom metrics.
        """
        if torch is None:
            return {"eval_loss": 0.0}

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        loader_kwargs = {"batch_size": self.batch_size, "shuffle": False}
        if self.data_collator:
            loader_kwargs["collate_fn"] = self.data_collator

        loader = DataLoader(eval_dataset, **loader_kwargs)

        total_loss = 0.0
        total_steps = 0
        all_predictions = []
        all_labels = []

        start_time = time.time()

        with torch.no_grad():
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            for batch in loader:
                # Move batch to device
                input_batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                with torch.amp.autocast(device_type=device_type):
                    outputs = model(**input_batch)

                    # Compute loss
                    if loss_fn is not None:
                        try:
                            loss = loss_fn(outputs)
                        except Exception:
                            loss = (
                                outputs.loss
                                if hasattr(outputs, "loss")
                                else outputs.sum()
                            )
                    else:
                        loss = (
                            outputs.loss if hasattr(outputs, "loss") else outputs.sum()
                        )

                total_loss += loss.item()
                total_steps += 1

                # Collect predictions/labels for custom metrics
                if hasattr(outputs, "logits"):
                    all_predictions.append(outputs.logits.detach().cpu())
                if "labels" in input_batch:
                    all_labels.append(input_batch["labels"].detach().cpu())

        eval_time = time.time() - start_time
        avg_loss = total_loss / max(total_steps, 1)

        metrics: Dict[str, float] = {
            "eval_loss": avg_loss,
            "eval_time_s": eval_time,
            "eval_samples": (
                len(eval_dataset)
                if hasattr(eval_dataset, "__len__")
                else total_steps * self.batch_size
            ),
        }

        # Perplexity (exp of loss — standard for language models)
        if compute_perplexity:
            try:
                metrics["eval_perplexity"] = math.exp(avg_loss)
            except OverflowError:
                metrics["eval_perplexity"] = float("inf")

        # Custom metrics
        if self.metrics_fns and all_predictions and all_labels:
            preds = torch.cat(all_predictions, dim=0)
            labels = torch.cat(all_labels, dim=0)
            for name, fn in self.metrics_fns.items():
                try:
                    metrics[f"eval_{name}"] = float(fn(preds, labels))
                except Exception as e:
                    logger.warning(f"Metric '{name}' failed: {e}")
                    metrics[f"eval_{name}"] = 0.0

        logger.info(
            f"Evaluation complete: loss={avg_loss:.4f} | "
            + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "eval_loss")
        )

        return metrics


# ──────────────────────────────────────────────────────────────────────
#  Built-in Metric Functions
# ──────────────────────────────────────────────────────────────────────


def accuracy_metric(predictions: "torch.Tensor", labels: "torch.Tensor") -> float:
    """Top-1 accuracy for classification / next-token prediction."""
    if predictions.dim() > 2:
        # (batch, seq_len, vocab_size) → argmax over vocab
        preds = predictions.argmax(dim=-1)
    elif predictions.dim() == 2:
        preds = predictions.argmax(dim=-1)
    else:
        preds = predictions

    # Mask out padding / ignore tokens (label == -100)
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    correct = (preds[mask] == labels[mask]).float().mean()
    return correct.item()


def top_k_accuracy_metric(k: int = 5):
    """Factory for top-k accuracy metric."""

    def _metric(predictions: "torch.Tensor", labels: "torch.Tensor") -> float:
        if predictions.dim() < 2:
            return 0.0
        top_k = predictions.topk(k, dim=-1).indices
        mask = labels != -100
        if mask.sum() == 0:
            return 0.0
        labels_expanded = labels.unsqueeze(-1).expand_as(top_k)
        correct = (top_k[mask] == labels_expanded[mask]).any(dim=-1).float().mean()
        return correct.item()

    _metric.__name__ = f"top_{k}_accuracy"
    return _metric
