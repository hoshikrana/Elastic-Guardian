"""
EGX Parallel Advisor — Layer 3.

Advises on data / tensor / pipeline parallelism configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from egx.core.models import HardwareTopology


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    data_parallel_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    rationale: str


class ParallelAdvisor:
    """Recommends optimal parallel configuration."""

    def advise(self, topo: HardwareTopology, model_params: int) -> ParallelConfig:
        n = len(topo.gpus)
        if n <= 1:
            return ParallelConfig(1, 1, 1, "Single device — no parallelism.")

        per_gpu_vram = topo.gpus[0].vram_bytes if topo.gpus else 0
        param_bytes = model_params * 2  # FP16

        # Model fits on one GPU → pure data parallel
        if param_bytes < int(per_gpu_vram * 0.6):
            return ParallelConfig(n, 1, 1, f"Model fits. DP={n}.")

        # Model needs sharding
        if n <= 4:
            return ParallelConfig(1, n, 1, f"TP={n} for model sharding.")

        # Large cluster
        tp = min(4, n)
        dp = n // tp
        return ParallelConfig(dp, tp, 1, f"DP={dp}, TP={tp} for balanced throughput.")
