"""
EGX Data Prefetcher — Layer 5.

Overlaps data transfer to GPU with training computation on the next batch.
Ensures the accelerator is never stalled waiting for host-to-device IO.
"""

from __future__ import annotations

import torch
from typing import Any, Iterator


class GPUDataPrefetcher:
    """
    Zero-stall CUDA Prefetcher.
    Uses a dedicated CUDA stream for asynchronous transfers.
    """

    def __init__(self, loader: Iterator, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None
        self.next_batch = None
        self._preload()

    def _preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        if self.stream:
            with torch.cuda.stream(self.stream):
                self.next_batch = self._move_to_device(self.next_batch)
        else:
            self.next_batch = self._move_to_device(self.next_batch)

    def _move_to_device(self, batch: Any) -> Any:
        if isinstance(batch, dict):
            return {
                k: v.to(self.device, non_blocking=True) if hasattr(v, "to") else v
                for k, v in batch.items()
            }
        elif isinstance(batch, (list, tuple)):
            return [
                v.to(self.device, non_blocking=True) if hasattr(v, "to") else v
                for v in batch
            ]
        elif hasattr(batch, "to"):
            return batch.to(self.device, non_blocking=True)
        return batch

    def __next__(self) -> Any:
        if self.stream:
            torch.cuda.current_stream().wait_stream(self.stream)

        batch = self.next_batch
        if batch is None:
            raise StopIteration

        self._preload()
        return batch

    def __iter__(self):
        return self
