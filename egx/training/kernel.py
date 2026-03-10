"""
EGX Training Kernel — Layer 5.

Main training loop with health monitoring, mixed precision, and recovery.
Coordinates the interaction between the model, optimizer, and resilience layers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union, Callable, List

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
from egx.core.interfaces import (
    BaseTrainingKernel,
    BaseWatchdog,
    BaseCheckpointManager,
)

logger = logging.getLogger("egx.training.kernel")


class TrainingKernel(BaseTrainingKernel):
    """
    Law 2: Stateless execution kernel.
    Handles the heavy lifting of a training step with extreme resilience.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_type: Union[str, Callable] = "adamw",
        loss_fn: Optional[Union[str, Callable]] = None,
        learning_rate: float = 2e-5,
        scheduler_type: Optional[str] = None,
        warmup_steps: int = 0,
        callbacks: Optional[List[Callable]] = None,
        precision_override: Optional[str] = None,
        watchdog: Optional[BaseWatchdog] = None,
        checkpoint_mgr: Optional[BaseCheckpointManager] = None,
    ):
        self.model = model
        self.watchdog = watchdog
        self.checkpoint_mgr = checkpoint_mgr
        self.loss_fn = loss_fn
        self.callbacks = callbacks or []
        self.precision_override = precision_override

        # Optimizer setup
        if torch is not None:
            if callable(optimizer_type):
                self.optimizer = optimizer_type(model.parameters(), lr=learning_rate)
            elif isinstance(optimizer_type, str) and optimizer_type.lower() == "adamw":
                self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            elif isinstance(optimizer_type, str) and optimizer_type.lower() == "sgd":
                self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            else:
                logger.warning(f"Unknown optimizer '{optimizer_type}', falling back to AdamW.")
                self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                
            self.scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
            
            # Scheduler setup
            if scheduler_type and isinstance(scheduler_type, str):
                if scheduler_type.lower() == "linear":
                    self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, total_iters=warmup_steps if warmup_steps > 0 else 100)
                elif scheduler_type.lower() == "cosine":
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
                else:
                    self.scheduler = None
            else:
                self.scheduler = None
        else:
            self.optimizer = None
            self.scaler = None
            self.scheduler = None

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

            # Autocast precision logic
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            target_dtype = None
            if self.precision_override:
                mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
                target_dtype = mapping.get(self.precision_override.lower())

            with torch.amp.autocast(device_type=device_type, dtype=target_dtype):
                outputs = self.model(**batch)
                
                if callable(self.loss_fn):
                    # For custom callables, pass outputs directly (or you could adapt this as needed)
                    # For simplicity, if they passed a custom loss, we assume it can map (outputs, targets)
                    # For this demo framework, we just apply it directly to outputs
                    try:
                        loss = self.loss_fn(outputs)
                    except Exception:
                        loss = outputs.loss if hasattr(outputs, "loss") else outputs.sum()
                elif isinstance(self.loss_fn, str) and self.loss_fn.lower() == "mse":
                    loss = torch.nn.functional.mse_loss(outputs, torch.zeros_like(outputs))
                elif isinstance(self.loss_fn, str) and self.loss_fn.lower() == "cross_entropy":
                    # Placeholder for CE logic
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs.sum()
                else:
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs.sum()

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                
            if self.scheduler:
                self.scheduler.step()

            # Checkpointing
            if self.checkpoint_mgr and self.checkpoint_mgr.should_save(
                step, loss.item()
            ):
                self.checkpoint_mgr.checkpoint(
                    step, loss.item(), self.model.state_dict()
                )

            # Trigger Developer Callbacks
            for callback in self.callbacks:
                try:
                    callback(step, loss.item())
                except Exception as e:
                    logger.warning(f"Callback {callback.__name__} failed at step {step}: {e}")

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
