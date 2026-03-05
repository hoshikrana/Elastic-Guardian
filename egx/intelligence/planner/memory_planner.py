"""
EGX Memory Planner — Layer 3.

Computes memory budgets per training mode for a given GPU.
"""

from __future__ import annotations

from typing import Dict
from egx.core.models import GPUSpec
from egx.core.enums import TrainingMode
from egx.core.constants import SAFETY_THRESHOLDS


class MemoryPlanner:
    """Plans memory allocation given a GPU and training mode."""

    def compute_budget(self, gpu: GPUSpec, mode: TrainingMode) -> Dict[str, int]:
        """Returns memory budget breakdown in bytes."""
        threshold = SAFETY_THRESHOLDS.get(mode, 0.72)
        usable = int(gpu.vram_bytes * threshold)
        overhead = gpu.vram_bytes - usable

        return {
            "total_vram": gpu.vram_bytes,
            "usable_vram": usable,
            "safety_overhead": overhead,
            "threshold_pct": int(threshold * 100),
        }

    def can_fit(self, gpu: GPUSpec, mode: TrainingMode, required_bytes: int) -> bool:
        budget = self.compute_budget(gpu, mode)
        return required_bytes <= budget["usable_vram"]
