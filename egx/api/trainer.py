"""
EGX Trainer — Layer 7.

The definitive v2.0 Public API for EGX.
Provides a high-level, fully customizable interface for:
- Training with user-overridable training steps
- Evaluation with custom metrics
- Prediction / text generation
- Callback-driven lifecycle hooks
- Zero-config defaults that just work

Inspired by the trainer architectures of HuggingFace, PyTorch Lightning,
and the patterns used in production LLM training (Megatron, DeepSpeed).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Union

from egx.api.config import EGXConfig
from egx.api.callbacks import (
    TrainingCallback,
    CallbackHandler,
    LoggingCallback,
    EarlyStoppingCallback,
)
from egx.api.evaluator import EGXEvaluator
from egx.api.predictor import EGXPredictor
from egx.runtime.engine import EGXEngine
from egx.core.exceptions import EGXError

logger = logging.getLogger("egx.api.trainer")


class EGXTrainer:
    """
    Law 12: Frozen API contract (extended for v2.0).

    The primary entry point for all EGX operations.

    Flexible Design Patterns:
    1. **Zero-config**: Just call train(model, dataset) — everything is automatic.
    2. **Config-driven**: Pass EGXConfig or dict for fine-grained control.
    3. **Callbacks**: Register TrainingCallback subclasses for lifecycle hooks.
    4. **Custom training step**: Override the forward+backward logic entirely.
    5. **Custom metrics**: Supply compute_metrics_fn for evaluation.
    6. **Evaluation**: Call .evaluate() standalone or enable eval-during-training.
    7. **Prediction**: Call .predict() or .generate() for inference.

    Example:
        # Zero-config
        trainer = EGXTrainer()
        result = trainer.train(model, dataset)

        # Full control
        trainer = EGXTrainer(
            config=EGXConfig(num_epochs=5, batch_size=4, eval_strategy="epoch"),
            callbacks=[EarlyStoppingCallback(patience=3), LoggingCallback()],
            training_step_fn=my_custom_step,
            compute_metrics_fn=my_metrics,
        )
        result = trainer.train(model, train_data, eval_dataset=val_data)
        metrics = trainer.evaluate(eval_dataset=test_data)
        texts = trainer.generate(prompts=["Hello"], tokenizer=tok)
    """

    def __init__(
        self,
        config: Optional[Union[EGXConfig, Dict[str, Any]]] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        training_step_fn: Optional[Callable] = None,
        compute_metrics_fn: Optional[Callable] = None,
        data_collator: Optional[Callable] = None,
    ):
        """
        Initialize the trainer.

        Args:
            config: Training configuration (EGXConfig, dict, or None for defaults).
            callbacks: List of TrainingCallback instances for lifecycle hooks.
            training_step_fn: Custom function(model, batch, step) -> float loss.
                             If provided, completely replaces the default training step.
            compute_metrics_fn: Custom function(predictions, labels) -> Dict[str, float].
                               Called during evaluation to compute domain-specific metrics.
            data_collator: Custom collate function for the DataLoader.
        """
        if isinstance(config, dict):
            self.config = EGXConfig.from_dict(config)
        else:
            self.config = config or EGXConfig()

        self._engine = EGXEngine()
        self._is_booted = False
        self._model = None
        self._evaluator = EGXEvaluator(
            batch_size=self.config.eval_batch_size,
            data_collator=data_collator,
        )
        self._predictor = EGXPredictor()

        # User-supplied overrides
        self._custom_training_step = training_step_fn
        self._compute_metrics = compute_metrics_fn
        self._data_collator = data_collator

        # Callback setup
        self._callback_handler = CallbackHandler(callbacks)

        # Auto-add LoggingCallback if not already provided
        has_logging = any(
            isinstance(c, LoggingCallback) for c in self._callback_handler.callbacks
        )
        if not has_logging:
            self._callback_handler.add(
                LoggingCallback(log_every_n_steps=self.config.logging_steps)
            )

        # Auto-add EarlyStoppingCallback if patience > 0
        if self.config.early_stopping_patience > 0:
            has_es = any(
                isinstance(c, EarlyStoppingCallback)
                for c in self._callback_handler.callbacks
            )
            if not has_es:
                self._callback_handler.add(
                    EarlyStoppingCallback(
                        patience=self.config.early_stopping_patience,
                        min_delta=self.config.early_stopping_threshold,
                    )
                )

        # Fire on_init_end
        self._callback_handler.fire("on_init_end", trainer=self)

    def __repr__(self) -> str:
        return (
            f"EGXTrainer(config={self.config!r}, "
            f"callbacks={len(self._callback_handler.callbacks)}, "
            f"booted={self._is_booted})"
        )

    def train(
        self,
        model: Any,
        dataset: Any,
        eval_dataset: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Main training entry point.

        Executes the 10-phase definitive lifecycle with full callback
        support and optional user-overridden training step.

        Args:
            model: The model to train (nn.Module or HuggingFace model).
            dataset: Training dataset (list of dicts, torch Dataset, or HF Dataset).
            eval_dataset: Optional evaluation dataset.

        Returns:
            Dictionary with training results (final_loss, duration_s, etc.).
        """
        try:
            self._model = model
            logger.info(
                f"EGX v2.0: Starting training session for {type(model).__name__}"
            )

            # 1. Boot the Engine (Phases 1-4)
            if not self._is_booted:
                self._engine.boot(model, self.config)
                self._is_booted = True

            # 2. Execute Training Logic (Phases 5-9)
            result = self._engine.run_training(
                model=model,
                dataset=dataset,
                eval_dataset=eval_dataset,
                config=self.config,
                callback_handler=self._callback_handler,
                training_step_fn=self._custom_training_step,
                data_collator=self._data_collator,
                compute_metrics_fn=self._compute_metrics,
                trainer_ref=self,
                **kwargs,
            )

            # 3. Shutdown (Phase 10)
            logger.info("EGX v2.0: Training session completed successfully.")
            return result

        except EGXError as e:
            logger.error(f"EGX Failure: {e.message}. Action: {e.suggested_action}")
            raise
        except Exception as e:
            logger.critical(f"Unexpected System Failure: {e}")
            raise EGXError(
                message="Fatal system error during training execution.",
                recoverable=False,
            ) from e

    def evaluate(
        self,
        model: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        loss_fn: Optional[Callable] = None,
        metrics_fns: Optional[Dict[str, Callable]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Run standalone evaluation.

        Args:
            model: Model to evaluate (uses last trained model if None).
            eval_dataset: Evaluation dataset.
            loss_fn: Optional custom loss function.
            metrics_fns: Dict of {metric_name: callable(preds, labels) -> float}.

        Returns:
            Dictionary of evaluation metrics.
        """
        model = model or self._model
        if model is None:
            raise ValueError("No model provided and no model from previous training.")

        if metrics_fns:
            self._evaluator.metrics_fns = metrics_fns

        self._callback_handler.fire("on_evaluate_begin", trainer=self)

        metrics = self._evaluator.evaluate(
            model=model,
            eval_dataset=eval_dataset,
            loss_fn=loss_fn,
            **kwargs,
        )

        self._callback_handler.fire("on_evaluate_end", trainer=self, metrics=metrics)
        return metrics

    def predict(
        self,
        model: Optional[Any] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Run a forward pass for prediction.

        Args:
            model: Model (uses last trained model if None).
            inputs: Input tensors dict.

        Returns:
            Model outputs.
        """
        model = model or self._model
        if model is None:
            raise ValueError("No model provided.")

        self._callback_handler.fire("on_predict_begin", trainer=self)
        outputs = self._predictor.predict(model, inputs)
        self._callback_handler.fire("on_predict_end", trainer=self)
        return outputs

    def generate(
        self,
        model: Optional[Any] = None,
        prompts: Optional[Union[str, List[str]]] = None,
        tokenizer: Any = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate text from prompts.

        Args:
            model: Language model (uses last trained model if None).
            prompts: Input prompts.
            tokenizer: Tokenizer for encoding/decoding.

        Returns:
            List of generated text strings.
        """
        model = model or self._model
        if model is None:
            raise ValueError("No model provided.")
        if prompts is None:
            raise ValueError("No prompts provided.")

        return self._predictor.generate(
            model=model,
            prompts=prompts,
            tokenizer=tokenizer,
            **kwargs,
        )

    def add_callback(self, callback: TrainingCallback) -> None:
        """Add a callback to the trainer."""
        self._callback_handler.add(callback)

    def remove_callback(self, callback_type: type) -> None:
        """Remove all callbacks of a given type."""
        self._callback_handler.remove(callback_type)

    @property
    def model(self) -> Optional[Any]:
        """Access the current model."""
        return self._model

    @property
    def callback_handler(self) -> CallbackHandler:
        """Access the callback handler."""
        return self._callback_handler


# Alias for definitive user access
EGX = EGXTrainer
