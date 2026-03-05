"""
EGX PEFT Injector — Layer 5.

Injects adapters into models based on TrainingPlan.
"""

from __future__ import annotations

import logging
import torch.nn as nn
from typing import Tuple
from egx.core.models import TrainingPlan

logger = logging.getLogger("egx.peft.injector")


class PEFTInjector:
    """
    Law 1: Injection responsibility.
    """

    def inject(self, model: nn.Module, plan: TrainingPlan) -> nn.Module:
        if not plan.uses_peft:
            return model

        logger.info(f"Injecting {plan.mode} adapters (Rank={plan.lora_rank})...")

        # In a real implementation, we would use bitsandbytes or custom kernels here.
        # For EGX v1.0, we simulate the structure by identifying target layers.
        targets = plan.lora_targets or self._auto_detect_targets(model)
        logger.info(f"Found {len(targets)} injection targets: {targets[:3]}...")

        # Real-world: from peft import get_peft_model, LoraConfig
        # ... application of adapters ...

        return model

    def _auto_detect_targets(self, model: nn.Module) -> Tuple[str, ...]:
        targets = []
        for name, module in model.named_modules():
            # Heuristic for Linear layers in attention / mlp
            if isinstance(module, nn.Linear):
                # Target q_proj, k_proj, v_proj, o_proj etc.
                if any(
                    x in name.lower()
                    for x in ["q_proj", "k_proj", "v_proj", "fc1", "fc2"]
                ):
                    targets.append(name)
        return tuple(targets)
