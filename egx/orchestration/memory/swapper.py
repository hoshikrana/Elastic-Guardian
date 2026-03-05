"""
EGX Memory Swapper Hub — Layer 4.

High-level coordinator for bidirectional tensor movement between
VRAM (GPU), RAM (CPU), and NVMe (Disk).
"""

from __future__ import annotations

import logging

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from egx.orchestration.swapper.vram_to_ram import VRAMToRAMSwapper
from egx.orchestration.swapper.ram_to_nvme import RAMToNVMeSwapper

logger = logging.getLogger("egx.orchestration.swapper")


class Swapper:
    """
    EGX Swapper Hub.
    Coordinates specialized swappers to provide a unified memory management API.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.vram_to_ram = VRAMToRAMSwapper()
        self.ram_to_nvme = RAMToNVMeSwapper()

    def offload_to_ram(self, model: nn.Module, layer_prefix: str) -> int:
        """Offload model parameters from VRAM to system RAM."""
        return self.vram_to_ram.offload(model, layer_prefix)

    def restore_from_ram(self, model: nn.Module, layer_prefix: str) -> None:
        """Restore model parameters from system RAM to VRAM."""
        self.vram_to_ram.restore(model, layer_prefix, device=self.device)

    def offload_to_disk(self, name: str, tensor: torch.Tensor) -> int:
        """Offload a single tensor from RAM to NVMe disk."""
        return self.ram_to_nvme.offload(name, tensor)

    def restore_from_disk(self, name: str, device: str = "cpu") -> torch.Tensor:
        """Restore a single tensor from NVMe disk to specified device."""
        return self.ram_to_nvme.restore(name, device=device)

    def cleanup(self) -> None:
        """Clear all disk-based caches."""
        self.ram_to_nvme.cleanup()

    @property
    def vram_offloaded_count(self) -> int:
        return self.vram_to_ram.offloaded_count

    @property
    def nvme_cached_count(self) -> int:
        return self.ram_to_nvme.cached_count
