"""
EGX Hardware Simulator — Testing Utility.

Mocks hardware capabilities for core logic verification.
Uses v1.0 definitive contracts (GPUSpec, HardwareTopology).
"""

from __future__ import annotations

from egx.core.enums import InterconnectType
from egx.core.models import GPUSpec, HardwareTopology


def mock_8gb_gpu() -> HardwareTopology:
    """Mock a single 8GB consumer GPU (RTX 3070-like)."""
    gpu = GPUSpec(
        device_id=0,
        name="Mock RTX 3070",
        vram_bytes=8 * 1024 * 1024 * 1024,
        compute_capability=(8, 6),
        memory_bandwidth_gbps=448.0,
        fp16_tflops=20.3,
        bf16_tflops=20.3,
        supports_flash_attn2=True,
        supports_fp8=False,
        nvlink_peer_ids=(),
    )

    return HardwareTopology(
        gpus=(gpu,),
        cpu_cores=8,
        ram_bytes=32 * 1024 * 1024 * 1024,
        nvme_bytes=500 * 1024 * 1024 * 1024,
        nvme_seq_read_gbps=3.5,
        nvme_seq_write_gbps=2.5,
        pcie_bandwidth_gbps=15.8,
        gpu_interconnect_gbps=15.8,
        interconnect=InterconnectType.PCIE,
        node_count=1,
    )


def mock_h100_cluster() -> HardwareTopology:
    """Mock a dual H100 setup."""
    gpus = tuple(
        GPUSpec(
            device_id=i,
            name="Mock H100",
            vram_bytes=80 * 1024 * 1024 * 1024,
            compute_capability=(9, 0),
            memory_bandwidth_gbps=3350.0,
            fp16_tflops=989.0,
            bf16_tflops=989.0,
            supports_flash_attn2=True,
            supports_fp8=True,
            nvlink_peer_ids=(1 - i,),
        )
        for i in range(2)
    )

    return HardwareTopology(
        gpus=gpus,
        cpu_cores=64,
        ram_bytes=512 * 1024 * 1024 * 1024,
        nvme_bytes=4000 * 1024 * 1024 * 1024,
        nvme_seq_read_gbps=7.0,
        nvme_seq_write_gbps=5.0,
        pcie_bandwidth_gbps=63.0,
        gpu_interconnect_gbps=900.0,
        interconnect=InterconnectType.NVLINK,
        node_count=1,
    )
