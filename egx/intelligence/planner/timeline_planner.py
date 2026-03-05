"""
EGX Timeline Planner — Layer 3.

Estimates training time based on hardware throughput and model size.
"""

from __future__ import annotations

from dataclasses import dataclass
from egx.core.models import GPUSpec


@dataclass(frozen=True, slots=True)
class TimelineEstimate:
    total_steps: int
    seconds_per_step: float
    estimated_total_seconds: float
    estimated_hours: float


class TimelinePlanner:
    """Estimates wall-clock training time."""

    def estimate(
        self,
        total_tokens: int,
        seq_length: int,
        batch_size: int,
        gpu: GPUSpec,
        num_gpus: int = 1,
    ) -> TimelineEstimate:
        tokens_per_step = batch_size * seq_length * num_gpus
        total_steps = max(1, total_tokens // tokens_per_step)

        # Rough throughput model: TFLOPS → tokens/sec
        tflops = gpu.fp16_tflops
        # Heuristic: ~1000 tokens/s per TFLOP for transformer training
        tokens_per_sec = tflops * 800 * num_gpus
        secs_per_step = tokens_per_step / max(1.0, tokens_per_sec)
        total_secs = total_steps * secs_per_step

        return TimelineEstimate(
            total_steps=total_steps,
            seconds_per_step=round(secs_per_step, 3),
            estimated_total_seconds=round(total_secs, 1),
            estimated_hours=round(total_secs / 3600, 2),
        )
