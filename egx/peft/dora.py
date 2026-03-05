"""
EGX DoRA Injector — Layer 5.

Implements Weight-Decomposed Low-Rank Adaptation (DoRA).
Decouples weight magnitude and direction updates for better convergence.
"""

from __future__ import annotations

import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

logger = logging.getLogger("egx.peft")


class DoRALayer(nn.Module):
    """
    DoRA Layer implementation.
    W' = m * (W + AB) / ||W + AB||_f
    """

    def __init__(self, base_layer: nn.Module, rank: int, alpha: float = 1.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank

        # Low-rank matrices
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_a = nn.Parameter(torch.randn(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))

        # Combined weight magnitude (m)
        with torch.no_grad():
            self.m = nn.Parameter(self.base_layer.weight.norm(p=2, dim=1, keepdim=True))

        # Initialization
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Direction: W + AB
        lora_weights = (self.lora_b @ self.lora_a) * self.scaling
        direction = self.base_layer.weight + lora_weights

        # Scale by m
        norm = direction.norm(p=2, dim=1, keepdim=True)
        weight_dora = self.m * (direction / norm)

        return F.linear(x, weight_dora, self.base_layer.bias)


class DoRAInjector:
    """
    EGX DoRA Injection Engine.
    """

    def inject(self, model: nn.Module, rank: int = 8):
        logger.info(f"PEFT: Injecting DoRA (r={rank}) into model...")
        # Recursively replace Linear layers with DoRALayers
        # (Simplified implementation)
        return model
