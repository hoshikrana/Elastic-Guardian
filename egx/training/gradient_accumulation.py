"""
EGX Virtual Batching — Layer 5.

Implements gradient accumulation to enable training large models
on small-memory devices without losing gradient fidelity.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("egx.training.accumulation")


class GradientAccumulator:
    """
    Manages micro-batching for constrained systems.
    Calculates step boundaries and scaling factors.
    """

    __slots__ = ("target_steps", "current_step", "total_accumulated")

    def __init__(self, target_steps: int):
        self.target_steps = max(1, target_steps)
        self.current_step = 0
        self.total_accumulated = 0

    def __repr__(self) -> str:
        return f"GradientAccumulator(target={self.target_steps}, current={self.current_step})"

    def should_step(self) -> bool:
        """Returns True if normalized gradients should be applied now."""
        self.current_step += 1
        ready = self.current_step >= self.target_steps
        if ready:
            self.current_step = 0
            self.total_accumulated += 1
        return ready

    def get_scale(self) -> float:
        """The loss must be divided by the accumulation factor."""
        return 1.0 / self.target_steps
