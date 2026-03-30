"""
EGX LoRA Weight Merger — Layer 5.

Critical logic for merging adapters into base weights for deployment.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("egx.peft")


class LoRAMerger:
    """
    Weights Merge Engine.
    Implements W = W + (A * B) * (alpha / rank).
    """

    def merge_weights(self, model: Any):
        """
        Zero-copy (where possible) weight merge.
        """
        import torch
        from egx.peft.lora import LoRALinear

        logger.info("PEFT: Starting LoRA weights merge...")
        count = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, LoRALinear):
                # Compute W = W_base + (B @ A) * scaling
                with torch.no_grad():
                    lora_update = (module.lora_B @ module.lora_A) * module.scaling
                    module.original.weight.data += lora_update

                # Replace LoRALinear with the original nn.Linear in the parent
                parent_name, attr_name = (
                    name.rsplit(".", 1) if "." in name else ("", name)
                )
                parent = (
                    dict(model.named_modules())[parent_name] if parent_name else model
                )
                setattr(parent, attr_name, module.original)
                count += 1

        logger.info(
            f"PEFT: Merge complete. {count} adapters fully integrated into base."
        )
        return model
