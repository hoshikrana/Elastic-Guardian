"""
EGX Prefetch Executor — Layer 4.

Prefetches tensors from RAM/NVMe to VRAM ahead of forward pass.
"""

from __future__ import annotations

import logging
from typing import List
import torch
import torch.nn as nn

logger = logging.getLogger("egx.orchestration.executor")


class PrefetchExecutor:
    """Async prefetcher that moves tensors to GPU before they're needed."""

    def __init__(self):
        self._stream = None
        if torch.cuda.is_available():
            self._stream = torch.cuda.Stream()

    def prefetch(self, model: nn.Module, layer_names: List[str]) -> None:
        """Prefetch specified layers to VRAM asynchronously."""
        if self._stream is None:
            self._sync_prefetch(model, layer_names)
            return

        with torch.cuda.stream(self._stream):
            for name, param in model.named_parameters():
                if any(ln in name for ln in layer_names):
                    if not param.is_cuda:
                        param.data = param.data.cuda(non_blocking=True)

    def sync(self) -> None:
        """Wait for prefetch to complete."""
        if self._stream is not None:
            self._stream.synchronize()

    def _sync_prefetch(self, model: nn.Module, layer_names: List[str]) -> None:
        for name, param in model.named_parameters():
            if any(ln in name for ln in layer_names):
                if torch.cuda.is_available() and not param.is_cuda:
                    param.data = param.data.cuda()
