"""
EGX Metrics Registry — Layer 4.

Centralized store for training telemetry (Loss, VRAM, Throughput).
Law 2: No global mutable state — each registry is an isolated instance.
"""

from __future__ import annotations

from typing import Dict, Optional
from collections import deque


class MetricRegistry:
    """
    Lock-free metric aggregation.

    Each trainer owns its own instance — no shared singleton state.
    """

    def __init__(self):
        self.metrics: Dict[str, deque] = {}

    def log(self, name: str, value: float, window: int = 100):
        """Records a metric with a rolling window."""
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=window)
        self.metrics[name].append(value)

    def get_avg(self, name: str) -> float:
        """Returns the average value in the current window."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return sum(self.metrics[name]) / len(self.metrics[name])

    def get_last(self, name: str) -> Optional[float]:
        """Returns the most recent value."""
        if name not in self.metrics or not self.metrics[name]:
            return None
        return self.metrics[name][-1]

    def reset(self):
        """Clears all stored metrics."""
        self.metrics.clear()

    def __repr__(self) -> str:
        return f"MetricRegistry(metrics={list(self.metrics.keys())})"
