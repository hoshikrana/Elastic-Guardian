"""
EGX Gradient Checkpointing Plugin — Layer 5.

Enables activation checkpointing to trade compute for memory.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger("egx.plugins.gradient_checkpointing")


class GradientCheckpointingPlugin:
    """Enables gradient checkpointing on transformer blocks."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._patched_count = 0

    def apply(self, model: nn.Module) -> nn.Module:
        """Enable gradient checkpointing on the model."""
        if not self.enabled:
            return model

        # HuggingFace models
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled (HF API).")
            return model

        # Manual: wrap each transformer block
        for name, module in model.named_children():
            if self._is_transformer_block(module):
                self._wrap_block(model, name, module)
                self._patched_count += 1

        logger.info(f"Gradient checkpointing applied to {self._patched_count} blocks.")
        return model

    def _is_transformer_block(self, module: nn.Module) -> bool:
        name = type(module).__name__.lower()
        return any(k in name for k in ["block", "layer", "decoder", "encoder"])

    def _wrap_block(self, parent: nn.Module, name: str, block: nn.Module) -> None:
        original_forward = block.forward

        def checkpointed_forward(*args, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)
            return original_forward(*args, **kwargs)

        block.forward = checkpointed_forward

    @property
    def patched_count(self) -> int:
        return self._patched_count
