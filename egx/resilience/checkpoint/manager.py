"""
EGX Checkpoint Manager — Layer 4.

Adaptive checkpoint strategy implementation.
"""

from __future__ import annotations

import time
import logging
from egx.core.enums import CheckpointStrategy
from egx.resilience.checkpoint.writer import CheckpointWriter

logger = logging.getLogger("egx.resilience.checkpoint")


class CheckpointManager:
    """
    Law 4: Orchestrates checkpoint lifecycle.
    """

    def __init__(
        self,
        output_dir: str,
        strategy: CheckpointStrategy = CheckpointStrategy.ADAPTIVE,
    ):
        self.output_dir = output_dir
        self.strategy = strategy
        self.writer = CheckpointWriter()
        self._last_save_time = time.time()
        self._best_loss = float("inf")
        self._last_save_step = 0

    def should_save(self, step: int, loss: float) -> bool:
        if self.strategy == CheckpointStrategy.STEP_BASED:
            return step % 500 == 0

        if self.strategy == CheckpointStrategy.TIME_BASED:
            return (time.time() - self._last_save_time) > 1800  # 30 mins

        if self.strategy == CheckpointStrategy.LOSS_BASED:
            return loss < self._best_loss * 0.99

        # Adaptive (Default)
        time_elapsed = time.time() - self._last_save_time
        loss_improved = loss < self._best_loss * 0.99

        # Save if loss improved significantly, or if it's been too long
        return loss_improved or (time_elapsed > 3600)

    def checkpoint(self, step: int, loss: float, state_dict: dict):
        path = f"{self.output_dir}/checkpoint_step_{step}.pt"
        self.writer.save({"step": step, "loss": loss, "state_dict": state_dict}, path)

        if loss < self._best_loss:
            self._best_loss = loss

        self._last_save_time = time.time()
        self._last_save_step = step
