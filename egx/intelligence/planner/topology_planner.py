"""
EGX Topology Planner — Layer 3.

Decides parallelism strategy based on hardware topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from egx.core.models import HardwareTopology
from egx.core.enums import InterconnectType


@dataclass(frozen=True, slots=True)
class ParallelismPlan:
    strategy: str  # "single", "dp", "fsdp", "pipeline"
    num_gpus: int
    shard_factor: int
    rationale: str


class TopologyPlanner:
    """Selects parallelism strategy from hardware topology."""

    def plan(self, topo: HardwareTopology, model_bytes: int) -> ParallelismPlan:
        num_gpus = len(topo.gpus)

        if num_gpus <= 1:
            return ParallelismPlan("single", 1, 1, "Single GPU — no parallelism needed.")

        per_gpu = topo.gpus[0].vram_bytes if topo.gpus else 0
        fits_single = model_bytes < int(per_gpu * 0.7)

        if fits_single:
            return ParallelismPlan(
                "dp", num_gpus, 1,
                f"Model fits in single GPU. Using DataParallel across {num_gpus} GPUs."
            )

        if topo.interconnect == InterconnectType.NVLINK:
            return ParallelismPlan(
                "fsdp", num_gpus, num_gpus,
                f"NVLink detected. Using FSDP with {num_gpus}-way sharding."
            )

        return ParallelismPlan(
            "pipeline", num_gpus, num_gpus,
            f"PCIe interconnect. Using Pipeline Parallelism across {num_gpus} GPUs."
        )
