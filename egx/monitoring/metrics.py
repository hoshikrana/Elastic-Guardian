"""
EGX Metrics Registry — Layer 4.

Centralized store for training telemetry (Loss, VRAM, Throughput).
"""

from __future__ import annotations

from typing import Dict, Optional
from collections import deque


class MetricRegistry:
    """
    Lock-free metric aggregation.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricRegistry, cls).__new__(cls)
            cls._instance.metrics: Dict[str, deque] = {}
        return cls._instance

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
