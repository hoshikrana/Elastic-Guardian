"""
EGX Training Kernel — Layer 5.

Main training loop with health monitoring, mixed precision, and recovery.
Coordinates the interaction between the model, optimizer, and resilience layers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from egx.core.enums import RecoveryAction
from egx.core.exceptions import EGXError, OutOfMemoryError
from egx.resilience.watchdog import TrainingWatchdog
from egx.resilience.checkpoint.manager import CheckpointManager

logger = logging.getLogger("egx.training.kernel")


class TrainingKernel:
    """
    Law 2: Stateless execution kernel.
    Handles the heavy lifting of a training step with extreme resilience.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_type: str = "adamw",
        learning_rate: float = 2e-5,
        watchdog: Optional[TrainingWatchdog] = None,
        checkpoint_mgr: Optional[CheckpointManager] = None,
    ):
        self.model = model
        self.watchdog = watchdog
        self.checkpoint_mgr = checkpoint_mgr

        # Optimizer setup (simplified for v1.0)
        if torch is not None:
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            self.scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
        else:
            self.optimizer = None
            self.scaler = None

    def train_step(self, batch: Dict[str, Any], step: int) -> float:
        """
        Executes a single training step with recovery guards and mixed precision.
        """
        if self.watchdog:
            self.watchdog.heartbeat(step)

        if self.optimizer is None:
            logger.warning("No optimizer available. Skipping step.")
            return 0.0

        try:
            self.optimizer.zero_grad()

            # Use autocast if available
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.amp.autocast(device_type=device_type):
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs.sum()

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Checkpointing
            if self.checkpoint_mgr and self.checkpoint_mgr.should_save(
                step, loss.item()
            ):
                self.checkpoint_mgr.checkpoint(
                    step, loss.item(), self.model.state_dict()
                )

            return float(loss.item())

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                logger.error("VRAM Exhausted: Kernel triggering OOM protocol.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise OutOfMemoryError() from e
            raise EGXError(
                str(e), recoverable=True, suggested_action=RecoveryAction.RETRY
            ) from e

        except Exception as e:
            logger.error(f"Unexpected error in Kernel: {e}")
            raise EGXError(
                message=f"Kernel crash at step {step}: {e}",
                recoverable=True,
                suggested_action=RecoveryAction.RETRY,
            ) from e
