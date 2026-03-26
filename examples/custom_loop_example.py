#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EGX Custom Training Loop Example

Demonstrates the full flexibility of EGX v2.0 — users can:
1. Write their OWN training step (forward + backward logic)
2. Create custom callbacks with domain-specific hooks
3. Supply custom metrics for evaluation
4. Override any part of the pipeline while keeping EGX resilience

This is how frameworks like Megatron-LM and DeepSpeed allow users
to customize training while providing infrastructure benefits.
"""

import logging
import torch
import torch.nn as nn
from typing import Any, Dict

from egx.api.trainer import EGXTrainer
from egx.api.config import EGXConfig
from egx.api.callbacks import TrainingCallback, LoggingCallback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("custom_loop")


# ──────────────────────────────────────────────────────────────────────
#  1. Define your model (any PyTorch model works)
# ──────────────────────────────────────────────────────────────────────

class MyTransformerModel(nn.Module):
    """A simple transformer-like model for demonstration."""

    def __init__(self, vocab_size=1000, hidden=256, num_layers=4, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embedding(input_ids)
        x = self.encoder(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        # Return a simple namespace-like object
        return type("Outputs", (), {"loss": loss, "logits": logits})()


# ──────────────────────────────────────────────────────────────────────
#  2. Custom Training Step — FULL USER CONTROL
# ──────────────────────────────────────────────────────────────────────

def my_custom_training_step(model: nn.Module, batch: Dict[str, Any], step: int) -> float:
    """
    A completely user-defined training step.
    
    You control:
    - How the forward pass works
    - How the loss is computed
    - How gradients flow
    - What optimizations to apply
    - What to log
    
    EGX will call this instead of its built-in kernel.
    """
    # Get the optimizer (we'll store it as a model attribute for this demo)
    optimizer = getattr(model, "_optimizer", None)
    if optimizer is None:
        model._optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer = model._optimizer

    optimizer.zero_grad()

    # Forward
    outputs = model(**batch)
    loss = outputs.loss

    if loss is None:
        return 0.0

    # Optional: custom loss scaling, regularization, etc.
    # loss = loss + 0.01 * sum(p.pow(2.0).sum() for p in model.parameters())

    # Backward
    loss.backward()

    # Optional: gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Step
    optimizer.step()

    return float(loss.item())


# ──────────────────────────────────────────────────────────────────────
#  3. Custom Callbacks — Hook into ANY lifecycle event
# ──────────────────────────────────────────────────────────────────────

class LossPlateauDetector(TrainingCallback):
    """Custom callback: detect when loss plateaus and log a warning."""

    def __init__(self, window: int = 50, threshold: float = 0.001):
        self.window = window
        self.threshold = threshold
        self.losses: list = []

    def on_step_end(self, trainer, step, loss, **kwargs):
        self.losses.append(loss)
        if len(self.losses) >= self.window * 2:
            recent = sum(self.losses[-self.window:]) / self.window
            previous = sum(self.losses[-2 * self.window:-self.window]) / self.window

            if abs(recent - previous) < self.threshold:
                logger.warning(
                    f"⚠ Loss plateau detected at step {step}: "
                    f"Δ={abs(recent - previous):.6f}"
                )


class GradientStatsCallback(TrainingCallback):
    """Custom callback: monitor gradient statistics like in Megatron-LM."""

    def __init__(self, log_every: int = 100):
        self.log_every = log_every

    def on_step_end(self, trainer, step, **kwargs):
        if step > 0 and step % self.log_every == 0:
            model = getattr(trainer, "_model", None)
            if model is None:
                return

            total_norm = 0.0
            param_count = 0
            max_grad = 0.0

            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    max_grad = max(max_grad, p.grad.data.abs().max().item())
                    param_count += 1

            total_norm = total_norm ** 0.5
            logger.info(
                f"Gradient Stats @ step {step}: "
                f"norm={total_norm:.4f} | max={max_grad:.6f} | params={param_count}"
            )


# ──────────────────────────────────────────────────────────────────────
#  4. Custom Metrics — Domain-specific evaluation
# ──────────────────────────────────────────────────────────────────────

def custom_metrics_fn(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Custom metrics function that EGX calls during evaluation.
    
    You receive the raw predictions and labels and can compute
    whatever metrics your use case requires.
    """
    # Top-1 accuracy
    preds = predictions.argmax(dim=-1)
    mask = labels != -100
    if mask.sum() > 0:
        accuracy = (preds[mask] == labels[mask]).float().mean().item()
    else:
        accuracy = 0.0

    return {
        "accuracy": accuracy,
        "total_predictions": float(mask.sum().item()),
    }


# ──────────────────────────────────────────────────────────────────────
#  5. Main — Put it all together
# ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("EGX Custom Training Loop Demo")

    # Create model and datasets
    model = MyTransformerModel(vocab_size=500, hidden=128, num_layers=2, num_heads=4)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Dummy datasets (replace with your data)
    seq_len = 64
    train_dataset = [
        {
            "input_ids": torch.randint(0, 500, (seq_len,)),
            "labels": torch.randint(0, 500, (seq_len,)),
        }
        for _ in range(200)
    ]
    eval_dataset = [
        {
            "input_ids": torch.randint(0, 500, (seq_len,)),
            "labels": torch.randint(0, 500, (seq_len,)),
        }
        for _ in range(50)
    ]

    # Configure EGX
    config = EGXConfig(
        num_epochs=5,
        batch_size=8,
        learning_rate=1e-3,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        logging_steps=5,
        early_stopping_patience=3,
        output_dir="./egx_custom_output",
    )

    # Create trainer with FULL customization
    trainer = EGXTrainer(
        config=config,

        # YOUR custom training step — replaces the built-in kernel
        training_step_fn=my_custom_training_step,

        # YOUR custom metrics — called during evaluation
        compute_metrics_fn=custom_metrics_fn,

        # YOUR custom callbacks — hook into any lifecycle event
        callbacks=[
            LoggingCallback(log_every_n_steps=5),
            LossPlateauDetector(window=20, threshold=0.001),
            GradientStatsCallback(log_every=25),
        ],
    )

    # Train
    logger.info("Starting training with custom loop...")
    result = trainer.train(
        model=model,
        dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info(f"Training complete! Loss: {result['final_loss']:.4f}")
    logger.info(f"Steps: {result['global_steps']} | Time: {result['duration_s']:.1f}s")

    # Standalone evaluation
    logger.info("Running standalone evaluation...")
    eval_result = trainer.evaluate(eval_dataset=eval_dataset)
    logger.info(f"Eval Loss: {eval_result['eval_loss']:.4f}")
    for k, v in eval_result.items():
        if k != "eval_loss":
            logger.info(f"  {k}: {v:.4f}")

    logger.info("Custom training loop demo complete!")


if __name__ == "__main__":
    main()
