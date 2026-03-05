"""
EGX Automated Mixed Precision — Layer 5.

Selects and configures the optimal dtypes for weights, gradients,
and activations based on accelerator capabilities (BF16, FP16, Int8).
"""

from __future__ import annotations

from typing import Tuple
from egx.core.enums import DType
from egx.core.models import HardwareTopology


class PrecisionSelector:
    """
    Universal Precision Manager.
    Ensures 'zero-config' precision that just works on any hardware.
    """

    @staticmethod
    def select_optimal(topology: HardwareTopology) -> Tuple[DType, bool]:
        """
        Returns (WeightDType, UseAutocast).

        Logic:
        - Ampere+ NVIDIA: BF16 (Best stability/speed)
        - Old NVIDIA: FP16
        - Apple Silicon: BF16
        - CPU: FP32 (Default) or BF16 if supported
        """
        if not topology.gpus:
            return DType.FP32, False

        gpu = topology.gpus[0]

        # Ampere+ (compute capability >= 8.0) supports BF16
        if gpu.compute_capability[0] >= 8:
            return DType.BF16, True

        # Older CUDA GPUs
        if gpu.compute_capability[0] >= 7:
            return DType.FP16, True

        # Default fallback
        return DType.FP32, False
