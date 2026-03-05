"""
EGX VRAM-to-RAM Swapper — Layer 4.

Offloads tensors from GPU VRAM to system RAM.
"""

from __future__ import annotations

import logging
from typing import Dict
import torch
import torch.nn as nn

logger = logging.getLogger("egx.orchestration.swapper")


class VRAMToRAMSwapper:
    """Offloads model parameters from VRAM to pinned CPU memory."""

    def __init__(self):
        self._offloaded: Dict[str, torch.Tensor] = {}

    def offload(self, model: nn.Module, layer_prefix: str) -> int:
        """Offload all parameters matching prefix. Returns bytes freed."""
        freed = 0
        for name, param in model.named_parameters():
            if name.startswith(layer_prefix) and param.is_cuda:
                size = param.data.nelement() * param.data.element_size()
                self._offloaded[name] = param.data.cpu().pin_memory()
                param.data = torch.empty(0, device="cpu")
                freed += size
        if freed:
            torch.cuda.empty_cache()
            logger.info(f"Offloaded {layer_prefix}: {freed} bytes freed from VRAM")
        return freed

    def restore(
        self, model: nn.Module, layer_prefix: str, device: str = "cuda"
    ) -> None:
        """Restore offloaded parameters back to VRAM."""
        for name, param in model.named_parameters():
            if name in self._offloaded and name.startswith(layer_prefix):
                param.data = self._offloaded.pop(name).to(device, non_blocking=True)

    @property
    def offloaded_count(self) -> int:
        return len(self._offloaded)
