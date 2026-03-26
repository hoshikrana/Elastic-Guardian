"""
EGX GPU Probe — Layer 2.

Chain: NVML -> nvidia-smi -> /proc fallback.
Law 6: torch allowed here.
Law 10: vram returned in int bytes.
"""

from __future__ import annotations

import logging
import pynvml
import torch
from typing import List, Tuple, Any
from egx.core.device import get_default_device
from egx.core.models import GPUSpec, HardwareTier
from egx.core.interfaces import BaseGPUProber

logger = logging.getLogger("egx.infrastructure.gpu")

# Approximate performance metrics lookup (Bandwidth GB/s, FP16 TFLOPS, BF16 TFLOPS)
GPU_SPECS_LOOKUP = {
    "RTX 4090": {"bw": 1008.0, "fp16": 82.6, "bf16": 82.6},
    "RTX 3090": {"bw": 936.0, "fp16": 35.6, "bf16": 35.6},
    "A100": {"bw": 1555.0, "fp16": 312.0, "bf16": 312.0},
    "H100": {"bw": 3350.0, "fp16": 989.0, "bf16": 989.0},
    "RTX 4080": {"bw": 716.0, "fp16": 48.7, "bf16": 48.7},
    "RTX 3080": {"bw": 760.0, "fp16": 29.8, "bf16": 29.8},
}

class GPUProber(BaseGPUProber):
    """
    Production-grade GPU prober.
    Uses NVML with fallback to Torch and MPS.
    """

    __slots__ = ("_nvml_initialized",)

    def __init__(self):
        self._nvml_initialized = False

    def __enter__(self) -> GPUProber:
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
        except Exception as e:
            logger.warning("NVML Initialization failed: %s", e)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception as e:
                logger.error("NVML Shutdown failed: %s", e)
            finally:
                self._nvml_initialized = False

    def __del__(self):
        """Ensure NVML is properly shut down."""
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def probe(self) -> List[GPUSpec]:
        """Probes all available GPUs via the best available method."""
        gpus = []

        # Method 1: NVML (Best for topology/bandwidth)
        if self._nvml_initialized:
            try:
                gpus = self._probe_nvml()
            except Exception as e:
                logger.error("NVML Probe failed: %s. Trying fallback.", e)

        # Method 2: Torch (Fallback for basic stats)
        if not gpus and get_default_device() == "cuda":
            gpus = self._probe_torch()

        # Method 3: Apple Silicon (MPS)
        if not gpus and torch.backends.mps.is_available():
            gpus = self._probe_mps()

        if not gpus:
            logger.warning(
                "No compatible GPUs detected. Falling back to CPU-only mode (Law 9: Explicit)."
            )
            gpus = [self._cpu_spec()]

        return gpus

    def _cpu_spec(self) -> GPUSpec:
        """Returns a virtual GPUSpec representing the system CPU for fallback."""
        return GPUSpec(
            device_id=-1,
            name="System CPU",
            vram_bytes=0,  # Law 10: 0 for CPU vram
            compute_capability=(0, 0),
            memory_bandwidth_gbps=50.0,
            fp16_tflops=1.0,
            bf16_tflops=1.0,
            supports_flash_attn2=False,
            supports_fp8=False,
            nvlink_peer_ids=(),
            vendor="cpu",
        )

    def _get_hw_metrics(self, name: str, cap_major: int) -> Tuple[float, float, float]:
        """Returns (bandwidth_gbps, fp16_tflops, bf16_tflops) from lookup or heuristic."""
        for key, stats in GPU_SPECS_LOOKUP.items():
            if key.lower() in name.lower():
                return stats["bw"], stats["fp16"], stats["bf16"]
        
        # Defaults based on compute capability
        if cap_major >= 9:
            return 2000.0, 500.0, 500.0
        elif cap_major >= 8:
            return 900.0, 50.0, 50.0
        else:
            return 400.0, 15.0, 15.0

    def _probe_nvml(self) -> List[GPUSpec]:
        gpus = []
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # Capability
            cap_major, cap_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)

            # Bandwidth (Approximation or fixed lookup)
            # For simplicity in this v1.0, we use compute capability to infer features
            supports_fa2 = cap_major >= 8  # Ampere+
            bw, fp16, bf16 = self._get_hw_metrics(name, cap_major)

            spec = GPUSpec(
                device_id=i,
                name=name,
                vram_bytes=int(mem_info.total),  # Law 10
                compute_capability=(cap_major, cap_minor),
                memory_bandwidth_gbps=bw,
                fp16_tflops=fp16,
                bf16_tflops=bf16,
                supports_flash_attn2=supports_fa2,
                supports_fp8=(cap_major >= 9),  # Hopper+
                nvlink_peer_ids=self._get_peer_ids(handle, device_count),
            )
            gpus.append(spec)
        return gpus

    def _probe_torch(self) -> List[GPUSpec]:
        logger.info("Falling back to Torch for GPU probing.")
        gpus = []
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory
            cap = torch.cuda.get_device_capability(i)
            bw, fp16, bf16 = self._get_hw_metrics(name, cap[0])

            spec = GPUSpec(
                device_id=i,
                name=name,
                vram_bytes=int(vram),
                compute_capability=cap,
                memory_bandwidth_gbps=bw,
                fp16_tflops=fp16,
                bf16_tflops=bf16,
                supports_flash_attn2=(cap[0] >= 8),
                supports_fp8=(cap[0] >= 9),
                nvlink_peer_ids=(),
            )
            gpus.append(spec)
        return gpus

    def _probe_mps(self) -> List[GPUSpec]:
        # Apple Silicon detection
        # MPS doesn't expose many stats directly via torch at runtime easily for VRAM total
        # We use a heuristic or sysctl if available
        return [
            GPUSpec(
                device_id=0,
                name="Apple M-Series (MPS)",
                vram_bytes=8 * 1024 * 1024 * 1024,  # Fixed placeholder, usually unified
                compute_capability=(0, 0),
                memory_bandwidth_gbps=200.0,
                fp16_tflops=10.0,
                bf16_tflops=10.0,
                supports_flash_attn2=False,
                supports_fp8=False,
                nvlink_peer_ids=(),
                vendor="apple",
            )
        ]

    def _get_peer_ids(self, handle: Any, count: int) -> Tuple[int, ...]:
        peers = []
        for j in range(count):
            try:
                if (
                    pynvml.nvmlDeviceGetP2PStatus(
                        handle, j, pynvml.NVML_P2P_CAPS_INDEX_READ
                    )
                    == pynvml.NVML_P2P_STATUS_OK
                ):
                    peers.append(j)
            except Exception:
                pass
        return tuple(peers)
