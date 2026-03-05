"""
EGX LoRA+ Adapter — Layer 5.

LoRA+ uses different learning rates for A and B matrices.
B gets a higher LR multiplier for faster convergence.
"""

from __future__ import annotations

import logging
from typing import List
import torch.nn as nn

logger = logging.getLogger("egx.peft.lora_plus")


def get_lora_plus_param_groups(
    model: nn.Module,
    base_lr: float = 2e-5,
    lora_b_multiplier: float = 16.0,
) -> List[dict]:
    """
    LoRA+ optimizer param groups.
    A matrices → base_lr, B matrices → base_lr * multiplier.
    """
    group_a = []
    group_b = []
    group_other = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name:
            group_a.append(param)
        elif "lora_B" in name:
            group_b.append(param)
        else:
            group_other.append(param)

    groups = []
    if group_a:
        groups.append({"params": group_a, "lr": base_lr, "name": "lora_A"})
    if group_b:
        groups.append(
            {"params": group_b, "lr": base_lr * lora_b_multiplier, "name": "lora_B"}
        )
    if group_other:
        groups.append({"params": group_other, "lr": base_lr, "name": "other"})

    logger.info(
        f"LoRA+ groups: A={len(group_a)} params (lr={base_lr}), "
        f"B={len(group_b)} params (lr={base_lr * lora_b_multiplier})"
    )
    return groups
