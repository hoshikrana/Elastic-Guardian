"""
EGX Throughput & MFU Tracker — Layer 5.

Measures tokens/sec and Model FLOPS Utilization (MFU).
Used for real-time performance optimization and training telemetry.
"""

from __future__ import annotations

import time
import logging
from typing import List

logger = logging.getLogger("egx.training.telemetry")


class ThroughputTracker:
    """
    Measures the 'heartbeat' of the training engine.
    """
    
    def __init__(self):
        self.start_time = None
        self.total_tokens = 0
        self.history: List[float] = []

    def start(self):
        self.start_time = time.time()

    def record_step(self, num_tokens: int):
        self.total_tokens += num_tokens
        
    @property
    def tokens_per_sec(self) -> float:
        if not self.start_time: return 0.0
        elapsed = time.time() - self.start_time
        return self.total_tokens / elapsed if elapsed > 0 else 0.0

    def get_mfu(self, total_flops: float, peak_flops: float) -> float:
        """Calculates Model FLOPS Utilization."""
        # Simple MFU calculation: (Achieved FLOPS / Peak FLOPS)
        if peak_flops <= 0: return 0.0
        return total_flops / peak_flops
