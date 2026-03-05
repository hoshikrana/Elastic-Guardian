"""
EGX Strategy Scorer — Layer 3.

Scores training strategies based on hardware fit, efficiency, and memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
from egx.core.models import GPUSpec
from egx.core.enums import TrainingMode
from egx.core.constants import SAFETY_THRESHOLDS


@dataclass(frozen=True, slots=True)
class ScoredStrategy:
    mode: TrainingMode
    score: float
    fits: bool
    rationale: str


class StrategyScorer:
    """Scores each training mode for a given hardware and model config."""

    def score_all(
        self,
        gpu: GPUSpec,
        model_bytes: int,
        modes: List[TrainingMode],
    ) -> List[ScoredStrategy]:
        results = []
        for mode in modes:
            threshold = SAFETY_THRESHOLDS.get(mode, 0.72)
            budget = int(gpu.vram_bytes * threshold)
            fits = model_bytes <= budget

            # Score formula: higher is better
            # Prefer modes that use more VRAM efficiently
            utilization = min(1.0, model_bytes / max(1, budget))
            quality_bonus = {
                TrainingMode.FULL_FINETUNE: 1.0,
                TrainingMode.LORA_PLUS: 0.85,
                TrainingMode.LORA: 0.80,
                TrainingMode.DORA: 0.82,
                TrainingMode.QLORA: 0.70,
            }.get(mode, 0.5)

            score = (utilization * 0.4 + quality_bonus * 0.6) if fits else 0.0

            results.append(ScoredStrategy(
                mode=mode,
                score=round(score, 4),
                fits=fits,
                rationale=f"Budget={budget}, Model={model_bytes}, Util={utilization:.2f}"
            ))

        results.sort(key=lambda s: s.score, reverse=True)
        return results
