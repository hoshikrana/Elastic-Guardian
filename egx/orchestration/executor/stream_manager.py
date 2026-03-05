"""
EGX Stream Manager — Layer 4.

Manages CUDA streams for overlapping compute and data movement.
"""

from __future__ import annotations

import logging
from typing import Optional
import torch

logger = logging.getLogger("egx.orchestration.executor")


class StreamManager:
    """Manages compute and transfer CUDA streams for overlap."""

    def __init__(self):
        self.compute_stream: Optional[torch.cuda.Stream] = None
        self.transfer_stream: Optional[torch.cuda.Stream] = None
        if torch.cuda.is_available():
            self.compute_stream = torch.cuda.Stream()
            self.transfer_stream = torch.cuda.Stream()

    def begin_transfer(self) -> Optional[torch.cuda.Stream]:
        return self.transfer_stream

    def begin_compute(self) -> Optional[torch.cuda.Stream]:
        return self.compute_stream

    def sync_all(self) -> None:
        if self.compute_stream:
            self.compute_stream.synchronize()
        if self.transfer_stream:
            self.transfer_stream.synchronize()

    @property
    def available(self) -> bool:
        return self.compute_stream is not None
