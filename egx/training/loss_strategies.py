"""
Loss Function Strategies — Layer 5.

Abstract strategy pattern for loss calculations.
Replaces inline branching and string matching with extensible, polymorphic design.

This eliminates:
- String matching on loss type ("mse", "cross_entropy", etc.)
- Complex if/elif branches in train_step()
- Runtime type dispatching

Enables:
- Easy addition of new loss types (just subclass LossFunctionStrategy)
- Clear separation of concerns
- Testable loss calculations in isolation
- Type-safe loss function handling
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

logger = logging.getLogger("egx.training.loss_strategies")


class LossFunctionStrategy(ABC):
    """
    Abstract base class for loss function strategies.

    Each strategy encapsulates a specific loss computation pattern.
    This allows users to extend with custom losses by subclassing.
    """

    @abstractmethod
    def compute(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute loss from model outputs and input batch.

        Args:
            outputs: Model output (typically from model(**batch))
            batch: Input batch dict (may contain 'labels', 'target', etc.)

        Returns:
            torch.Tensor: Scalar loss value

        Raises:
            ValueError: If required fields missing or incompatible types
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """Readable strategy name."""
        pass


class CallableLossStrategy(LossFunctionStrategy):
    """
    Strategy for callable loss functions (user-provided functions).

    Handles any Python callable that takes (outputs, batch) -> loss.
    Includes fallback if callable expects only outputs.
    """

    def __init__(self, fn: Callable):
        if not callable(fn):
            raise TypeError(f"Expected callable, got {type(fn)}")
        self.fn = fn

    def compute(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        try:
            # Try with both outputs and batch
            loss = self.fn(outputs, batch)
        except TypeError:
            # Fallback: try with outputs only
            try:
                loss = self.fn(outputs)
            except Exception as e:
                # Last resort: check for outputs.loss attribute
                if hasattr(outputs, "loss"):
                    loss = outputs.loss
                elif hasattr(outputs, "sum"):
                    loss = outputs.sum()
                else:
                    logger.error(f"Callable loss function failed: {e}")
                    raise ValueError(
                        f"Loss callable failed and no fallback available: {e}"
                    ) from e

        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"Loss function must return torch.Tensor, got {type(loss)}")

        return loss

    def __repr__(self) -> str:
        name = getattr(self.fn, "__name__", str(self.fn))
        return f"CallableLoss({name})"


class HFModelDefaultLossStrategy(LossFunctionStrategy):
    """
    Strategy for HuggingFace models with built-in loss calculation.

    When model is passed (inputs, labels), it returns output.loss.
    This is the default for transformers.PreTrainedModel derivatives.
    """

    def compute(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        if not hasattr(outputs, "loss"):
            raise ValueError(
                f"Model outputs missing 'loss' attribute. "
                f"Ensure model is HuggingFace style or has loss calculation. "
                f"Got outputs: {type(outputs)}"
            )

        loss = outputs.loss
        if not isinstance(loss, torch.Tensor):
            raise TypeError(
                f"Expected outputs.loss to be torch.Tensor, got {type(loss)}"
            )

        return loss

    def __repr__(self) -> str:
        return "HFModelDefaultLoss(outputs.loss)"


class MSELossStrategy(LossFunctionStrategy):
    """
    Strategy for Mean Squared Error loss.

    Requires 'labels' in batch. Applied as: MSE(outputs, batch['labels'])
    Useful for regression tasks.
    """

    def compute(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        if "labels" not in batch:
            raise ValueError(
                "MSE loss requires 'labels' in batch. "
                f"Available keys: {list(batch.keys())}"
            )

        labels = batch["labels"]

        # Handle outputs that might be a dict or nested structure
        if isinstance(outputs, dict) and "logits" in outputs:
            outputs = outputs["logits"]

        loss = F.mse_loss(outputs, labels)

        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"F.mse_loss returned {type(loss)}, expected torch.Tensor")

        return loss

    def __repr__(self) -> str:
        return "MSELoss(F.mse_loss)"


class CrossEntropyLossStrategy(LossFunctionStrategy):
    """
    Strategy for Cross-Entropy loss (classification).

    Requires 'labels' in batch. Applied as: CE(outputs, batch['labels'])
    Useful for multi-class classification tasks.
    """

    def compute(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        if "labels" not in batch:
            raise ValueError(
                "Cross-Entropy loss requires 'labels' in batch. "
                f"Available keys: {list(batch.keys())}"
            )

        labels = batch["labels"]

        # Handle outputs that might be a dict with 'logits' key
        if isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs

        loss = F.cross_entropy(logits, labels)

        if not isinstance(loss, torch.Tensor):
            raise TypeError(
                f"F.cross_entropy returned {type(loss)}, expected torch.Tensor"
            )

        return loss

    def __repr__(self) -> str:
        return "CrossEntropyLoss(F.cross_entropy)"


class BCEWithLogitsLossStrategy(LossFunctionStrategy):
    """
    Strategy for Binary Cross-Entropy with Logits loss.

    Requires 'labels' in batch. Applied as: BCE(outputs, batch['labels'])
    Useful for binary classification or multi-label tasks.
    """

    def compute(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        if "labels" not in batch:
            raise ValueError(
                "BCE loss requires 'labels' in batch. "
                f"Available keys: {list(batch.keys())}"
            )

        labels = batch["labels"]

        if isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs

        loss = F.binary_cross_entropy_with_logits(logits, labels)

        if not isinstance(loss, torch.Tensor):
            raise TypeError(
                f"F.binary_cross_entropy_with_logits returned {type(loss)}, expected torch.Tensor"
            )

        return loss

    def __repr__(self) -> str:
        return "BCEWithLogitsLoss(F.binary_cross_entropy_with_logits)"


class SumLossStrategy(LossFunctionStrategy):
    """
    Fallback strategy: sum all values in outputs.

    Used when loss function is None and model doesn't have outputs.loss.
    This is a last-resort strategy for debugging/prototyping.
    """

    def compute(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            loss = outputs.sum()
        elif isinstance(outputs, dict):
            # Try to find a 'loss' key
            if "loss" in outputs:
                loss = outputs["loss"]
            else:
                # Sum all tensors in the dict
                tensors = [v for v in outputs.values() if isinstance(v, torch.Tensor)]
                if not tensors:
                    raise ValueError(
                        "No tensors found in outputs dict for sum loss fallback"
                    )
                loss = sum(t.sum() for t in tensors)
        else:
            raise ValueError(f"Cannot compute sum loss for output type {type(outputs)}")

        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(loss)}")

        return loss

    def __repr__(self) -> str:
        return "SumLoss(outputs.sum())"


class LossFunctionFactory:
    """
    Factory for creating loss function strategies.

    Handles:
    - String names ("mse", "cross_entropy", etc.)
    - Callable functions (user-provided)
    - TrainingMode enums with loss specifications
    - None (defaults to HF model loss)

    Eliminates string matching from train_step() entirely.
    """

    # Mapping of string names to strategy classes
    _strategy_map = {
        "mse": MSELossStrategy,
        "mean_squared_error": MSELossStrategy,
        "cross_entropy": CrossEntropyLossStrategy,
        "ce": CrossEntropyLossStrategy,
        "categorical_crossentropy": CrossEntropyLossStrategy,
        "bce": BCEWithLogitsLossStrategy,
        "binary_cross_entropy": BCEWithLogitsLossStrategy,
        "bce_with_logits": BCEWithLogitsLossStrategy,
        "hf_default": HFModelDefaultLossStrategy,
    }

    @staticmethod
    def create(loss_fn: Optional[Union[str, Callable]] = None) -> LossFunctionStrategy:
        """
        Create a loss strategy from various input types.

        Args:
            loss_fn: One of:
                - None: Use HF model default loss (outputs.loss)
                - str: Strategy name ("mse", "cross_entropy", etc.)
                - Callable: User-provided loss function

        Returns:
            LossFunctionStrategy: Appropriate strategy instance

        Raises:
            ValueError: If strategy name is unknown
            TypeError: If input type is unsupported
        """
        if loss_fn is None:
            logger.info("Loss function not specified, using HF model default loss")
            return HFModelDefaultLossStrategy()

        if isinstance(loss_fn, str):
            loss_name = loss_fn.lower().strip()
            if loss_name not in LossFunctionFactory._strategy_map:
                available = ", ".join(LossFunctionFactory._strategy_map.keys())
                raise ValueError(
                    f"Unknown loss function: '{loss_fn}'. " f"Available: {available}"
                )
            strategy_class = LossFunctionFactory._strategy_map[loss_name]
            logger.info(f"Creating loss strategy: {strategy_class.__name__}")
            return strategy_class()

        if callable(loss_fn):
            logger.info(f"Creating callable loss strategy: {loss_fn.__name__}")
            return CallableLossStrategy(loss_fn)

        raise TypeError(
            f"Expected loss_fn to be None, str, or callable. Got {type(loss_fn)}"
        )

    @staticmethod
    def register_strategy(name: str, strategy_class: type) -> None:
        """
        Register a custom loss strategy.

        Allows users to add new loss types without modifying this file.

        Args:
            name: Name to register under (e.g., "custom_huber")
            strategy_class: Subclass of LossFunctionStrategy
        """
        if not issubclass(strategy_class, LossFunctionStrategy):
            raise TypeError(
                f"strategy_class must be subclass of LossFunctionStrategy, "
                f"got {strategy_class}"
            )
        name_lower = name.lower().strip()
        LossFunctionFactory._strategy_map[name_lower] = strategy_class
        logger.info(
            f"Registered loss strategy: {name_lower} -> {strategy_class.__name__}"
        )

    @staticmethod
    def available_strategies() -> Dict[str, type]:
        """Get dict of available strategies."""
        return LossFunctionFactory._strategy_map.copy()
