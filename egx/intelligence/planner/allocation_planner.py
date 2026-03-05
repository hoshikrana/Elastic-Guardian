"""
EGX Allocation Planner — Layer 3.

Generates the proactive AllocationPlan roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

from egx.core.enums import TrainingMode


@dataclass(frozen=True, slots=True)
class AllocationPlan:
    """Roadmap for tensor placement."""
    vram_tensors: List[str] = field(default_factory=list)
    ram_tensors: List[str] = field(default_factory=list)
    prefetch_schedule: Dict[str, List[str]] = field(default_factory=dict)


class AllocationPlanner:
    """Generates the proactive roadmap."""

    def plan(self, mode: TrainingMode) -> AllocationPlan:
        if mode in (TrainingMode.LORA, TrainingMode.LORA_PLUS, TrainingMode.DORA):
            return AllocationPlan(
                vram_tensors=["adapters", "activations"],
                ram_tensors=["base_weights"],
                prefetch_schedule={"pre_forward": ["base_weights"]}
            )
        if mode == TrainingMode.QLORA:
            return AllocationPlan(
                vram_tensors=["quantized_weights", "adapters"],
                ram_tensors=[],
                prefetch_schedule={}
            )
        return AllocationPlan(vram_tensors=["all"])
