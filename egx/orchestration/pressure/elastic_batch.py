"""
EGX Elastic Batch Resizer — Layer 4.

Dynamically resizes batch size based on VRAM pressure.
"""

from __future__ import annotations

import logging
import torch

logger = logging.getLogger("egx.orchestration.pressure")


class ElasticBatchResizer:
    """Halves or doubles batch size based on VRAM usage."""

    def __init__(self, initial_batch: int = 8, min_batch: int = 1, max_batch: int = 256):
        self.current_batch = initial_batch
        self.min_batch = min_batch
        self.max_batch = max_batch
        self._oom_count = 0

    def on_oom(self) -> int:
        """Called when OOM occurs. Halves the batch size."""
        self._oom_count += 1
        self.current_batch = max(self.min_batch, self.current_batch // 2)
        logger.warning(f"OOM #{self._oom_count}: Batch reduced to {self.current_batch}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.current_batch

    def try_increase(self, vram_usage_pct: float) -> int:
        """Attempts to increase batch size if pressure is low."""
        if vram_usage_pct < 0.6 and self.current_batch < self.max_batch:
            self.current_batch = min(self.max_batch, self.current_batch * 2)
            logger.info(f"Low pressure ({vram_usage_pct:.0%}): Batch increased to {self.current_batch}")
        return self.current_batch

    @property
    def oom_count(self) -> int:
        return self._oom_count
