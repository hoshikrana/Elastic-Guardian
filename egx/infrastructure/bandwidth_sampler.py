"""
EGX Bandwidth Sampler — Layer 2.

Measures actual PCIe and NVLink GB/s.
"""

from __future__ import annotations

import time
import torch
import logging

logger = logging.getLogger("egx.infrastructure.bandwidth")


class BandwidthSampler:
    """
    Law 22: Returns int bytes/s (or GB/s as float for spec).
    """

    def sample_pcie(self, device_id: int) -> float:
        """Measures CPU <-> GPU bandwidth."""
        device = torch.device(f"cuda:{device_id}")
        size = 1024 * 1024 * 512  # 512MB
        host_tensor = torch.randn(size, dtype=torch.float32).pin_memory()

        # Warmup
        for _ in range(3):
            _ = host_tensor.to(device, non_blocking=False)

        # Sync and measure
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(5):
            _ = host_tensor.to(device, non_blocking=False)
        torch.cuda.synchronize(device)
        total_time = time.perf_counter() - start

        gbps = (size * 4 * 5 / total_time) / 1e9  # 4 bytes per float32
        logger.info(f"GPU {device_id} PCIe Bandwidth: {gbps:.2f} GB/s")
        return gbps

    def sample_nvlink(self, src: int, dst: int) -> float:
        """Measures GPU <-> GPU bandwidth."""
        if src == dst:
            return 1000.0  # Virtual local loopback

        # Heuristic if p2p not enabled
        if not torch.cuda.can_device_access_peer(src, dst):
            return 0.0

        src_dev = torch.device(f"cuda:{src}")
        dst_dev = torch.device(f"cuda:{dst}")
        size = 1024 * 1024 * 512
        tensor = torch.randn(size, device=src_dev)

        torch.cuda.synchronize(src_dev)
        torch.cuda.synchronize(dst_dev)
        start = time.perf_counter()
        for _ in range(5):
            _ = tensor.to(dst_dev)
        torch.cuda.synchronize(src_dev)
        torch.cuda.synchronize(dst_dev)
        total_time = time.perf_counter() - start

        gbps = (size * 4 * 5 / total_time) / 1e9
        logger.info(f"GPU {src}->{dst} Interconnect: {gbps:.2f} GB/s")
        return gbps
