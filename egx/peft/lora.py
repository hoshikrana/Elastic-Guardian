"""
EGX LoRA Adapter — Layer 5.

Implements Low-Rank Adaptation for parameter-efficient fine-tuning.
"""

from __future__ import annotations

import math
import logging
from typing import Optional
import torch
import torch.nn as nn

from dataclasses import dataclass, field
 
logger = logging.getLogger("egx.peft.lora")
 
 
@dataclass(slots=True)
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[list[str]] = None
 
 
class LoRAModel(nn.Module):
    """Wrapper for a model with LoRA adapters."""
    
    __slots__ = ("base_model", "config")
 
    def __init__(self, model: nn.Module, config: LoRAConfig):
        super().__init__()
        self.base_model = inject_lora(
            model, 
            rank=config.rank, 
            alpha=config.alpha, 
            targets=config.target_modules
        )
        self.config = config
 
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
 
    def state_dict(self, *args, **kwargs):
        # We only want to save the LoRA weights in a real production scenario, 
        # but for this framework, we return the whole thing or filter as needed.
        return self.base_model.state_dict(*args, **kwargs)

class LoRALinear(nn.Module):
    """
    LoRA-injected Linear layer.
    W_new = W_frozen + (B @ A) * scaling
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        in_features = original.in_features
        out_features = original.out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)

        # Kaiming init for A, zero for B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

    @property
    def trainable_params(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


def inject_lora(
    model: nn.Module, rank: int = 16, alpha: int = 32, targets: Optional[list] = None
) -> nn.Module:
    """Inject LoRA adapters into target Linear layers."""
    target_names = targets or ["q_proj", "v_proj", "k_proj", "o_proj"]
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_names):
            parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = dict(model.named_modules())[parent_name] if parent_name else model
            setattr(parent, attr_name, LoRALinear(module, rank, alpha))
            count += 1
    logger.info(f"Injected LoRA into {count} layers (rank={rank}, alpha={alpha})")
    return model
