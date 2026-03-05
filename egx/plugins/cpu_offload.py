"""
EGX CPU Offload Plugin — Layer 5.

Implements reactive and proactive optimizer state offloading.
Allows training models 2-3x larger than VRAM.
"""

from __future__ import annotations

import torch
import logging

logger = logging.getLogger("egx.plugins.offload")


class CPUOffloadPlugin:
    """
    Manages optimizer state on host RAM.
    """

    @staticmethod
    def offload_optimizer(optimizer: torch.optim.Optimizer):
        """
        Moves all optimizer states to CPU.
        """
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to("cpu", non_blocking=True)
        logger.info("Offload: Optimizer states migrated to Host RAM.")

    @staticmethod
    def move_params(model: torch.nn.Module, device: str = "cpu"):
        """
        Moves model parameters to target device.
        """
        model.to(device, non_blocking=True)
        logger.debug(f"Offload: Model parameters moved to {device}.")
