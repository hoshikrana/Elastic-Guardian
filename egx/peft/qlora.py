"""
EGX QLoRA Adapter — Layer 5.

Quantized LoRA: 4-bit base weights + LoRA adapters.
"""

from __future__ import annotations

import logging
from typing import Optional
import torch
import torch.nn as nn

logger = logging.getLogger("egx.peft.qlora")


class QuantizedLinear(nn.Module):
    """
    Simulated 4-bit quantized linear with LoRA.
    In production, this delegates to bitsandbytes.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
    ):
        super().__init__()
        in_f = original.in_features
        out_f = original.out_features

        # Simulated 4-bit: store FP16 weights but mark as "quantized"
        self.weight_quantized = nn.Parameter(
            original.weight.data.half(), requires_grad=False
        )
        self.bias = original.bias

        # LoRA on top
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize (simulated)
        w = self.weight_quantized.float()
        base = nn.functional.linear(x, w, self.bias)
        lora = x @ self.lora_A.T @ self.lora_B.T
        return base + lora * self.scaling


def inject_qlora(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    targets: Optional[list] = None,
) -> nn.Module:
    """Inject QLoRA adapters into Linear layers."""
    target_names = targets or ["q_proj", "v_proj", "k_proj", "o_proj"]
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_names):
            parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = dict(model.named_modules())[parent_name] if parent_name else model
            setattr(parent, attr_name, QuantizedLinear(module, rank, alpha))
            count += 1
    logger.info(f"Injected QLoRA into {count} layers (rank={rank}, 4-bit)")
    return model
