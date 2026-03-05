"""
EGX Allocation Executor — Layer 4.

Executes the tensor placement plan from AllocationPlanner.
"""

from __future__ import annotations

import logging
from typing import List
import torch
import torch.nn as nn

logger = logging.getLogger("egx.orchestration.executor")


class AllocationExecutor:
    """Moves tensors between VRAM, RAM, and NVMe according to the plan."""

    def execute(self, model: nn.Module, plan_vram: List[str], plan_ram: List[str]) -> None:
        """Place model parameters according to the allocation plan."""
        for name, param in model.named_parameters():
            if self._matches(name, plan_ram):
                param.data = param.data.cpu()
                param.requires_grad = False
                logger.debug(f"Offloaded to RAM: {name}")
            elif self._matches(name, plan_vram):
                if torch.cuda.is_available():
                    param.data = param.data.cuda()
                logger.debug(f"Placed on VRAM: {name}")

    def _matches(self, name: str, patterns: List[str]) -> bool:
        if "all" in patterns:
            return True
        return any(p in name for p in patterns)
