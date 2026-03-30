"""
EGX Training Kernel — Layer 5.

Main training loop with health monitoring, mixed precision, and recovery.
Coordinates the interaction between the model, optimizer, and resilience layers.
"""

from __future__ import annotations

import inspect
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
from egx.core.device import get_default_device, get_device_type
from egx.training.loss_strategies import LossFunctionFactory, LossFunctionStrategy
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

    __slots__ = (
        "model",
        "watchdog",
        "checkpoint_mgr",
        "loss_fn",
        "loss_strategy",
        "callbacks",
        "precision_override",
        "max_grad_norm",
        "optimizer",
        "scaler",
        "scheduler",
    )

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
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.watchdog = watchdog
        self.checkpoint_mgr = checkpoint_mgr
        self.loss_fn = loss_fn
        # Create loss strategy using factory (eliminates string matching from train_step)
        self.loss_strategy = LossFunctionFactory.create(loss_fn)
        self.callbacks = callbacks or []
        self.precision_override = precision_override
        self.max_grad_norm = max_grad_norm

        # Optimizer setup
        if torch is not None:
            _OPTIMIZER_REGISTRY = {
                "adamw": lambda params, lr: torch.optim.AdamW(params, lr=lr),
                "sgd": lambda params, lr: torch.optim.SGD(params, lr=lr),
                "adafactor": lambda params, lr: torch.optim.AdamW(params, lr=lr),  # placeholder
            }

            if callable(optimizer_type):
                self.optimizer = optimizer_type(model.parameters(), lr=learning_rate)
            elif isinstance(optimizer_type, str):
                factory = _OPTIMIZER_REGISTRY.get(optimizer_type.lower())
                if factory:
                    self.optimizer = factory(model.parameters(), learning_rate)
                else:
                    logger.warning("Unknown optimizer '%s', falling back to AdamW.", optimizer_type)
                    self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            else:
                logger.warning("Invalid optimizer_type, falling back to AdamW.")
                self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                
            self.scaler = torch.amp.GradScaler() if get_device_type() == "cuda" else None
            
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

    def train_step(
        self,
        batch: Dict[str, Any],
        step: int,
        loss_scale: float = 1.0,
        should_optimizer_step: bool = True,
        accelerator: Any = None,
    ) -> float:
        """
        Executes a single training step with recovery guards and mixed precision.

        Args:
            batch: Input batch dict.
            step: Current global step.
            loss_scale: Scale factor applied to loss before backward (for grad accumulation).
            should_optimizer_step: If True, runs optimizer.step() + zero_grad() after backward.
                                  Set to False for intermediate accumulation micro-batches.
        """
        if self.watchdog:
            self.watchdog.heartbeat(step)

        if self.optimizer is None:
            logger.warning("No optimizer available. Skipping step.")
            return 0.0

        try:
            if should_optimizer_step:
                self.optimizer.zero_grad()

            # Autocast precision logic
            device_type = get_device_type()
            target_dtype = None
            if self.precision_override:
                mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
                target_dtype = mapping.get(self.precision_override.lower())

            with torch.amp.autocast(device_type=device_type, dtype=target_dtype):
                # ── Model Forward ──
                # Use introspection to see if 'labels' is supported (or if it takes **kwargs)
                sig = inspect.signature(self.model.forward)
                has_labels = "labels" in sig.parameters
                has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

                if "labels" in batch and not has_labels and not has_kwargs:
                    model_inputs = {k: v for k, v in batch.items() if k != "labels"}
                    if len(model_inputs) == 1:
                        outputs = self.model(next(iter(model_inputs.values())))
                    else:
                        outputs = self.model(**model_inputs)
                else:
                    outputs = self.model(**batch)

                # ── Loss Calculation (Now via strategy pattern) ──
                # This replaces 15+ lines of if/elif branching with a single call
                loss = self.loss_strategy.compute(outputs, batch)

            # Backpropagation and Optimization
            if accelerator:
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if self.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
            else:
                # Fallback path if accelerator is not provided
                scaled_loss = loss * loss_scale
                if self.scaler:
                    self.scaler.scale(scaled_loss).backward()
                    if should_optimizer_step:
                        if self.max_grad_norm > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    scaled_loss.backward()
                    if should_optimizer_step:
                        if self.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()

                if should_optimizer_step and self.scheduler:
                    self.scheduler.step()

            # Checkpointing
            if self.checkpoint_mgr and self.checkpoint_mgr.should_save(
                step, loss.item()
            ):
                self.checkpoint_mgr.checkpoint(
                    step, loss.item(), self.model.state_dict()
                )

            # Trigger Developer Callbacks
            loss_val = float(loss.item()) if hasattr(loss, "item") else float(loss)
            for callback in self.callbacks:
                try:
                    callback(step, loss_val)
                except Exception as e:
                    logger.warning(f"Callback {callback.__name__} failed at step {step}: {e}")

            return loss_val

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

    def __repr__(self) -> str:
        opt_name = type(self.optimizer).__name__ if self.optimizer else "None"
        return f"TrainingKernel(optimizer={opt_name}, max_grad_norm={self.max_grad_norm})"
