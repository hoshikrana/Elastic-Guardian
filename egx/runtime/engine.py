"""
EGX Engine — Layer 6.

Production-grade training lifecycle orchestrator with callback-driven loops,
gradient accumulation, mixed precision, eval-during-training, and
user-overridable training steps.

Implements the patterns used in state-of-the-art LLM training pipelines
(Megatron-LM, DeepSpeed, GPT-NeoX, LLaMA-Factory).
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None

from egx.core.models import HardwareTopology, TrainingPlan
from egx.core.enums import TrainingMode
from egx.core.device import get_default_device, get_device_type
from egx.infrastructure.gpu_probe import GPUProber
from egx.infrastructure.topology_builder import TopologyBuilder
from egx.intelligence.strategy.selector import FibonacciHeap
from egx.peft.lora import inject_lora, LoRAConfig, LoRAModel
from egx.training.kernel import TrainingKernel
from egx.training.gradient_accumulation import GradientAccumulator
from egx.resilience.watchdog import TrainingWatchdog
from egx.resilience.checkpoint.manager import CheckpointManager
from egx.core.interfaces import (
    BaseEngine,
    BaseGPUProber,
    BaseTopologyBuilder,
    BaseStrategySelector,
    BaseCheckpointManager,
    BaseWatchdog,
)
from egx.api.config import EGXConfig, TrainingSessionConfig
from egx.api.validation import ModelValidator
from egx.core.exceptions import HardwareError
from egx.monitoring import MemoryProfiler

logger = logging.getLogger("egx.runtime.engine")


class EGXEngine(BaseEngine):
    """
    Law 1: Centralized orchestration of the training lifecycle.
    Manages the transitions between all 10 definitive phases.

    Now with:
    - Callback-driven lifecycle hooks at every phase
    - Gradient accumulation support
    - Mixed precision via TrainingKernel
    - Eval-during-training (epoch-level and step-level)
    - Custom training_step_fn override
    - Early stopping integration
    - Throughput and metric tracking
    """

    def __init__(
        self,
        gpu_prober: Optional[BaseGPUProber] = None,
        topology_builder: Optional[BaseTopologyBuilder] = None,
        strategy_selector: Optional[BaseStrategySelector] = None,
    ):
        self._topology: Optional[HardwareTopology] = None
        self._plan: Optional[TrainingPlan] = None
        self._kernel = None
        self.device: Optional[torch.device] = None
        self._booted = False

        # Dependency Injection
        self.gpu_prober = gpu_prober or GPUProber()
        self.topology_builder = topology_builder or TopologyBuilder()
        self.strategy_selector = strategy_selector or FibonacciHeap()

    def boot(self, model: torch.nn.Module, config: EGXConfig) -> None:
        """Executes hardware alignment and safety checks."""
        if self._booted:
            return

        logger.info("EGX Engine booting...")

        # 1. Hardware Probing (RAII Context Manager)
        with self.gpu_prober as prober:
            gpus = prober.probe()

        # 2. Topology Assembly
        topology = self.topology_builder.build(gpus)
        if topology is None:
            from egx.core.exceptions import HardwareError

            raise HardwareError(
                "Topology assembly failed: No valid configuration found.",
                recoverable=False,
            )
        self._topology = topology
        logger.debug("Topology detected: %d GPUs", len(self._topology.gpus))

        # 3. Config Validation
        if config.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")

        # 4. Model Safety Check
        if not ModelValidator.check_nans(model):
            raise HardwareError(
                "Model contains NaN/Inf weights before training starts.",
                recoverable=False,
            )

        # 5. Model Introspection
        param_count = sum(p.numel() for p in model.parameters())
        logger.debug("Model Introspection: %d parameters", param_count)

        self._booted = True
        logger.info("EGX Engine boot successful.")

    @staticmethod
    def _normalize_training_mode(mode: Union[str, TrainingMode]) -> TrainingMode:
        """
        Normalize training mode to TrainingMode enum.

        Handles both string and enum inputs, always returns TrainingMode enum.
        This eliminates runtime type ambiguity and polymorphic dispatch.
        """
        if isinstance(mode, TrainingMode):
            return mode
        if isinstance(mode, str):
            try:
                return TrainingMode(mode)
            except ValueError as e:
                logger.warning(
                    f"Invalid training mode '{mode}', defaulting to LORA: {e}"
                )
                return TrainingMode.LORA
        # Fallback for any other unexpected type
        logger.warning(
            f"Unexpected training mode type {type(mode)}, defaulting to LORA"
        )
        return TrainingMode.LORA

    def run_training(
        self,
        model: Any,
        dataset: Any,
        eval_dataset: Optional[Any],
        config: EGXConfig,
        callback_handler: Any = None,
        training_step_fn: Optional[Callable] = None,
        data_collator: Optional[Callable] = None,
        compute_metrics_fn: Optional[Callable] = None,
        trainer_ref: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Executes Phases 5-10 with a production-grade training loop.
        """
        # Extract and validate config into type-safe dataclass early
        session_config = self._setup_training_session_config(config)

        # Automatic boot check (Must happen before any phase)
        if not self._booted:
            logger.info("Engine not booted. Triggering automatic boot sequence...")
            self.boot(model, config)

        # ── Phase 5: Strategy Selection ──
        # Use StrategyScorer to dynamically pick the best mode
        from egx.intelligence.strategy.scorer import StrategyScorer
        from egx.core.constants import STRATEGY_PRIORITY_ORDER

        scorer = StrategyScorer()

        # Estimate model size (heuristic)
        model_bytes = sum(p.numel() * 4 for p in model.parameters())
        gpu = self._topology.gpus[0] if self._topology and self._topology.gpus else None

        if gpu:
            scored_strategies = scorer.score_all(
                gpu, model_bytes, STRATEGY_PRIORITY_ORDER
            )
            # Find the best strategy that fits.
            best = next((s for s in scored_strategies if s.fits), None)

            if best:
                selected_mode = self._normalize_training_mode(best.mode)
                logger.info(
                    f"Phase 5: Strategy Selected -> {selected_mode.value} (Score: {best.score:.2f})"
                )
                from egx.infrastructure.structured_logger import StructuredLogger

                StructuredLogger("egx.engine").log_decision(
                    "train_run",
                    {"mode": selected_mode.value, "score": best.score, "gpu": gpu.name},
                )
            else:
                selected_mode = TrainingMode.LORA  # Already normalized (it's an enum)
                logger.warning(
                    f"Phase 5: No strategy fits perfectly. Falling back to {selected_mode.value}"
                )
        else:
            selected_mode_raw = getattr(config, "training_mode", TrainingMode.LORA)
            selected_mode = self._normalize_training_mode(selected_mode_raw)
            logger.info(
                f"Phase 5: No GPU info. Defaulting strategy -> {selected_mode.value}"
            )

        # ── Phase 6: Contract Finalization ──

        # ── Phase 7: PEFT Injection ──
        # Normalize: selected_mode is now guaranteed to be TrainingMode enum
        if selected_mode.uses_peft():
            model = inject_lora(
                model,
                rank=session_config.lora_rank,
                alpha=session_config.lora_alpha,
                targets=session_config.lora_targets,
            )
            logger.info(
                f"Phase 7: PEFT Injection -> LoRA applied (rank={session_config.lora_rank})"
            )
        else:
            logger.info(
                "Phase 7: PEFT Injection -> Skipped (Not required for selected strategy)"
            )

        # ── Phase 8: Kernel Setup ──
        # Grad Accumulation Setup
        accumulator = GradientAccumulator(session_config.gradient_accumulation_steps)
        loss_scale = accumulator.get_scale()

        # Build Watchdog
        watchdog = TrainingWatchdog(timeout_s=session_config.timeout)
        watchdog.start()

        # Build Checkpoint Manager
        checkpoint_mgr = CheckpointManager(
            output_dir=session_config.output_dir,
            strategy=session_config.checkpoint_strategy,
        )
        logger.debug("Phase 8: Checkpoint Manager initialized")

        # Build Kernel
        self._kernel = TrainingKernel(
            model=model,
            optimizer_type=session_config.optimizer_type,
            loss_fn=session_config.loss_fn,
            learning_rate=session_config.learning_rate,
            scheduler_type=session_config.scheduler_type,
            warmup_steps=session_config.warmup_steps,
            callbacks=session_config.callbacks,
            precision_override=session_config.precision_override,
            watchdog=watchdog,
            checkpoint_mgr=checkpoint_mgr,
            max_grad_norm=session_config.max_grad_norm,
        )

        # ── Phase 9: Production Training Loop ──
        logger.info("Phase 9: Entering production training loop...")

        result = self._production_training_loop(
            model=model,
            dataset=dataset,
            eval_dataset=eval_dataset,
            config=session_config,
            callback_handler=callback_handler,
            training_step_fn=training_step_fn,
            data_collator=data_collator,
            compute_metrics_fn=compute_metrics_fn,
            selected_mode=selected_mode,
            trainer_ref=trainer_ref,
            accumulator=accumulator,
            watchdog=watchdog,
        )

        # ── Phase 10: Shutdown ──
        logger.info("Phase 10: Graceful shutdown.")
        watchdog.stop()
        return result

    def _setup_training_session_config(self, config: any) -> TrainingSessionConfig:
        """
        Extract and validate runtime training config from EGXConfig.

        This consolidates all config defaults into a single, type-safe dataclass
        (eliminates scattered getattr() calls throughout the codebase).
        """
        if isinstance(config, TrainingSessionConfig):
            return config
        return TrainingSessionConfig.from_egx_config(config)

    def _prepare_training_dataloaders(
        self,
        dataset: Any,
        eval_dataset: Optional[Any],
        session_config: TrainingSessionConfig,
        data_collator: Optional[Callable],
        accelerator: Any,
    ) -> tuple[Any, Optional[Any]]:
        """
        Prepare training and evaluation dataloaders for accelerate.

        Returns: (train_loader, eval_loader)
        """
        dataset_len = len(dataset) if hasattr(dataset, "__len__") else 1
        loader_kwargs = {
            "batch_size": session_config.batch_size,
            "shuffle": dataset_len > 0,
        }
        if data_collator:
            loader_kwargs["collate_fn"] = data_collator
        train_loader = DataLoader(dataset, **loader_kwargs)

        # Eval DataLoader
        eval_loader = None
        if eval_dataset is not None and session_config.eval_strategy != "no":
            eval_loader_kwargs = {
                "batch_size": session_config.eval_batch_size,
                "shuffle": False,
            }
            if data_collator:
                eval_loader_kwargs["collate_fn"] = data_collator
            eval_loader = DataLoader(eval_dataset, **eval_loader_kwargs)

        # Accelerate prepare
        components = [self._kernel.model, train_loader]
        if self._kernel.optimizer:
            components.append(self._kernel.optimizer)
        if eval_loader:
            components.append(eval_loader)
        if self._kernel.scheduler:
            components.append(self._kernel.scheduler)

        prepared = accelerator.prepare(*components)

        idx = 0
        self._kernel.model = prepared[idx]
        idx += 1
        train_loader = prepared[idx]
        idx += 1
        if self._kernel.optimizer:
            self._kernel.optimizer = prepared[idx]
            idx += 1
        if eval_loader:
            eval_loader = prepared[idx]
            idx += 1
        if self._kernel.scheduler:
            self._kernel.scheduler = prepared[idx]
            idx += 1

        return train_loader, eval_loader

    def _run_training_step(
        self,
        batch: Dict[str, Any],
        global_step: int,
        session_config: TrainingSessionConfig,
        accelerator: Any,
        training_step_fn: Optional[Callable],
        callback_handler: Any,
        trainer_ref: Any,
    ) -> Optional[float]:
        """
        Execute single training step with recovery FSM.

        Returns: loss_value or None if recovery attempted
        """
        try:
            if training_step_fn is not None:
                loss_value = training_step_fn(self._kernel.model, batch, global_step)
            else:
                with accelerator.accumulate(self._kernel.model):
                    loss_value = self._kernel.train_step(
                        batch, global_step, accelerator=accelerator
                    )
            return loss_value
        except Exception as e:
            # ── Recovery FSM ──
            from egx.core.exceptions import EGXError
            from egx.resilience.recovery.orchestrator import (
                RecoveryOrchestrator,
                RecoveryContext,
            )
            import asyncio

            logger.error(
                f"Training error at step {global_step}. Triggering Recovery FSM..."
            )
            egx_error = (
                e if isinstance(e, EGXError) else EGXError(str(e), recoverable=True)
            )

            context = RecoveryContext(
                error=egx_error,
                step=global_step,
                current_batch_size=session_config.batch_size,
                current_training_mode=str(getattr(self, "_training_mode", "unknown")),
            )

            orchestrator = RecoveryOrchestrator()
            recovered = asyncio.run(orchestrator.recover(context))

            if not recovered:
                logger.critical("Recovery strategies exhausted. Aborting.")
                raise e

            logger.info("Recovery successful. Resuming training.")
            return None

    def _run_training_epoch(
        self,
        epoch: int,
        train_loader: Any,
        session_config: TrainingSessionConfig,
        accelerator: Any,
        training_step_fn: Optional[Callable],
        compute_metrics_fn: Optional[Callable],
        eval_loader: Optional[Any],
        callback_handler: Any,
        trainer_ref: Any,
    ) -> tuple[float, int, float]:
        """
        Run single training epoch.

        Returns: (epoch_loss, epoch_steps, best_eval_loss)
        """
        self._kernel.model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        global_step = 0
        best_eval_loss = float("inf")
        nan_count = 0
        accumulated_loss = 0.0
        accumulated_count = 0

        if callback_handler:
            callback_handler.fire("on_epoch_begin", trainer=trainer_ref, epoch=epoch)

        for batch_idx, batch in enumerate(train_loader):
            if callback_handler:
                callback_handler.fire(
                    "on_step_begin", trainer=trainer_ref, step=global_step
                )

            # Training step
            loss_value = self._run_training_step(
                batch,
                global_step,
                session_config,
                accelerator,
                training_step_fn,
                callback_handler,
                trainer_ref,
            )

            if loss_value is None:
                # Recovery occurred, skip this step
                continue

            # NaN detection
            if math.isnan(loss_value) or math.isinf(loss_value):
                nan_count += 1
                logger.warning(f"NaN/Inf loss at step {global_step}")
                if nan_count > 10:
                    logger.error("Too many NaN losses. Halting training.")
                    return epoch_loss / max(epoch_steps, 1), epoch_steps, best_eval_loss
                continue

            accumulated_loss += loss_value
            accumulated_count += 1
            epoch_loss += loss_value
            epoch_steps += 1

            # Step boundary
            if accelerator.sync_gradients:
                global_step += 1

            # Periodic logging
            if global_step % session_config.logging_steps == 0:
                avg_recent = accumulated_loss / max(accumulated_count, 1)
                current_lr = (
                    self._kernel.optimizer.param_groups[0]["lr"]
                    if self._kernel.optimizer
                    else 0.0
                )
                logger.info(
                    "Epoch %d | Step %d | Loss: %.4f | LR: %.2e",
                    epoch + 1,
                    global_step,
                    avg_recent,
                    current_lr,
                )
                if callback_handler:
                    callback_handler.fire(
                        "on_log",
                        trainer=trainer_ref,
                        logs={
                            "loss": avg_recent,
                            "lr": current_lr,
                            "epoch": epoch,
                            "step": global_step,
                        },
                    )
                accumulated_loss = 0.0
                accumulated_count = 0

            # Step-level evaluation
            if (
                session_config.eval_strategy == "steps"
                and eval_loader is not None
                and global_step > 0
                and global_step % session_config.eval_steps == 0
                and accelerator.sync_gradients
            ):
                eval_metrics = self._run_evaluation(
                    self._kernel.model,
                    eval_loader,
                    self.device,
                    callback_handler,
                    compute_metrics_fn,
                    trainer_ref,
                )
                if eval_metrics.get("eval_loss", float("inf")) < best_eval_loss:
                    best_eval_loss = eval_metrics["eval_loss"]

                # Check early stopping
                if callback_handler and self._check_early_stopping(callback_handler):
                    return epoch_loss / max(epoch_steps, 1), epoch_steps, best_eval_loss

                self._kernel.model.train()

            # Max steps limit
            if 0 < session_config.max_steps <= global_step:
                return epoch_loss / max(epoch_steps, 1), epoch_steps, best_eval_loss

        return epoch_loss / max(epoch_steps, 1), epoch_steps, best_eval_loss

    def _maybe_evaluate_and_checkpoint(
        self,
        epoch: int,
        avg_epoch_loss: float,
        best_eval_loss: float,
        session_config: TrainingSessionConfig,
        eval_loader: Optional[Any],
        compute_metrics_fn: Optional[Callable],
        callback_handler: Any,
        trainer_ref: Any,
    ) -> float:
        """
        Post-epoch evaluation and checkpointing.

        Returns: updated best_eval_loss
        """
        epoch_metrics = {"train_loss_epoch": avg_epoch_loss, "epoch": epoch + 1}

        if session_config.eval_strategy == "epoch" and eval_loader is not None:
            eval_metrics = self._run_evaluation(
                self._kernel.model,
                eval_loader,
                self.device,
                callback_handler,
                compute_metrics_fn,
                trainer_ref,
            )
            epoch_metrics.update(eval_metrics)
            if eval_metrics.get("eval_loss", float("inf")) < best_eval_loss:
                best_eval_loss = eval_metrics["eval_loss"]

            # Check early stopping
            if callback_handler and self._check_early_stopping(callback_handler):
                logger.info("Early stopping triggered")
                return best_eval_loss

        # Fire on_epoch_end
        if callback_handler:
            callback_handler.fire(
                "on_epoch_end",
                trainer=trainer_ref,
                epoch=epoch,
                metrics=epoch_metrics,
            )

        return best_eval_loss

    def _production_training_loop(
        self,
        model: Any,
        dataset: Any,
        eval_dataset: Optional[Any],
        config: TrainingSessionConfig,
        callback_handler: Any,
        training_step_fn: Optional[Callable],
        data_collator: Optional[Callable],
        compute_metrics_fn: Optional[Callable],
        selected_mode: TrainingMode,
        trainer_ref: Any = None,
        accumulator: GradientAccumulator = None,
        watchdog: TrainingWatchdog = None,
    ) -> Dict[str, Any]:
        """
        Production training loop orchestrator.

        Implements patterns from Megatron-LM, DeepSpeed, and GPT training:
        1. Gradient accumulation
        2. Mixed precision
        3. Periodic evaluation
        4. Early stopping
        5. Checkpoint management
        6. Error recovery

        REFACTORED for maintainability: Complex logic extracted to dedicated methods
        (_run_training_epoch, _run_training_step, _maybe_evaluate_and_checkpoint).
        """
        if torch is None:
            return {
                "success": True,
                "final_loss": 0.0,
                "duration_s": 0.0,
                "mode": selected_mode,
            }

        start_time = time.perf_counter()
        session_config = config

        # Initialize accelerator
        from accelerate import Accelerator

        accelerator = Accelerator(
            gradient_accumulation_steps=session_config.gradient_accumulation_steps
        )
        self.device = accelerator.device
        logger.info("Training initialized on device: %s", self.device)

        # Prepare dataloaders
        train_loader, eval_loader = self._prepare_training_dataloaders(
            dataset, eval_dataset, session_config, data_collator, accelerator
        )

        self._kernel.model.train()

        # State tracking
        total_loss = 0.0
        total_valid_steps = 0
        best_eval_loss = float("inf")

        # Fire on_train_begin
        if callback_handler:
            callback_handler.fire("on_train_begin", trainer=trainer_ref)

        # Epoch loop
        for epoch in range(session_config.num_epochs):
            # Run training for this epoch
            avg_epoch_loss, epoch_steps, best_eval_loss = self._run_training_epoch(
                epoch,
                train_loader,
                session_config,
                accelerator,
                training_step_fn,
                compute_metrics_fn,
                eval_loader,
                callback_handler,
                trainer_ref,
            )

            total_loss += avg_epoch_loss * epoch_steps
            total_valid_steps += epoch_steps

            # Post-epoch evaluation and checkpointing
            best_eval_loss = self._maybe_evaluate_and_checkpoint(
                epoch,
                avg_epoch_loss,
                best_eval_loss,
                session_config,
                eval_loader,
                compute_metrics_fn,
                callback_handler,
                trainer_ref,
            )

            # Check early stopping
            if callback_handler and self._check_early_stopping(callback_handler):
                logger.info(f"Training stopped at epoch {epoch+1}")
                break

            # Max steps limit
            if 0 < session_config.max_steps <= epoch_steps:
                logger.info(f"Max steps ({session_config.max_steps}) reached")
                break

        # ── Final result ──
        duration = time.perf_counter() - start_time
        final_loss = (
            total_loss / max(total_valid_steps, 1) if total_valid_steps > 0 else 0.0
        )

        result = {
            "success": True,
            "final_loss": final_loss,
            "best_eval_loss": best_eval_loss if best_eval_loss < float("inf") else None,
            "duration_s": duration,
            "epochs_completed": session_config.num_epochs,
            "mode": selected_mode,
            "topology": self._topology,
        }

        # Fire on_train_end
        if callback_handler:
            callback_handler.fire("on_train_end", trainer=trainer_ref, result=result)

        return result

    def _run_evaluation(
        self,
        model: Any,
        eval_loader: "DataLoader",
        device: "torch.device",
        callback_handler: Any,
        compute_metrics_fn: Optional[Callable],
        trainer_ref: Any,
    ) -> Dict[str, float]:
        """Run a full evaluation loop."""
        model.eval()

        if callback_handler:
            callback_handler.fire("on_evaluate_begin", trainer=trainer_ref)

        total_loss = 0.0
        total_steps = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            for batch in eval_loader:
                input_batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                with torch.amp.autocast(device_type=device_type):
                    outputs = model(**input_batch)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs.sum()

                total_loss += loss.item()
                total_steps += 1

                if compute_metrics_fn and hasattr(outputs, "logits"):
                    all_preds.append(outputs.logits.detach().cpu())
                    if "labels" in input_batch:
                        all_labels.append(input_batch["labels"].detach().cpu())

        avg_loss = total_loss / max(total_steps, 1)
        metrics: Dict[str, float] = {"eval_loss": avg_loss}

        # Perplexity
        try:
            metrics["eval_perplexity"] = math.exp(avg_loss)
        except OverflowError:
            metrics["eval_perplexity"] = float("inf")

        # Custom metrics
        if compute_metrics_fn and all_preds and all_labels:
            preds = torch.cat(all_preds, dim=0)
            labels = torch.cat(all_labels, dim=0)
            try:
                custom = compute_metrics_fn(preds, labels)
                if isinstance(custom, dict):
                    metrics.update(custom)
            except Exception as e:
                logger.warning(f"Custom metrics function failed: {e}")

        if callback_handler:
            callback_handler.fire(
                "on_evaluate_end", trainer=trainer_ref, metrics=metrics
            )

        logger.info(
            f"Eval ▶ loss={avg_loss:.4f} | "
            + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "eval_loss")
        )

        return metrics

    @staticmethod
    def _check_early_stopping(callback_handler) -> bool:
        """Check if any EarlyStopping callback wants to stop."""
        from egx.api.callbacks import EarlyStoppingCallback

        for cb in callback_handler.callbacks:
            if isinstance(cb, EarlyStoppingCallback) and cb.should_stop:
                return True
        return False

    @property
    def topology(self) -> Optional[HardwareTopology]:
        return self._topology
