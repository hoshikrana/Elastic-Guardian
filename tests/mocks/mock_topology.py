"""Reusable HardwareTopology presets for EGX test suite."""
from egx.core.models import HardwareTopology
from egx.core.enums import InterconnectType
from tests.mocks.mock_gpu import LAPTOP_GPU, A100_GPU, H100_GPU, CPU_ONLY


def laptop_topology() -> HardwareTopology:
    return HardwareTopology(
        gpus=(LAPTOP_GPU(),), cpu_cores=8, ram_bytes=16 * 1024**3,
        nvme_bytes=512 * 1024**3, nvme_seq_read_gbps=3.5, nvme_seq_write_gbps=2.5,
        pcie_bandwidth_gbps=15.8, gpu_interconnect_gbps=15.8,
        interconnect=InterconnectType.PCIE, node_count=1,
    )

def datacenter_topology() -> HardwareTopology:
    return HardwareTopology(
        gpus=(A100_GPU(), A100_GPU()), cpu_cores=64, ram_bytes=256 * 1024**3,
        nvme_bytes=2000 * 1024**3, nvme_seq_read_gbps=7.0, nvme_seq_write_gbps=5.0,
        pcie_bandwidth_gbps=31.5, gpu_interconnect_gbps=600.0,
        interconnect=InterconnectType.NVLINK, node_count=1,
    )

def cluster_topology() -> HardwareTopology:
    gpus = tuple(H100_GPU() for _ in range(4))
    return HardwareTopology(
        gpus=gpus, cpu_cores=128, ram_bytes=512 * 1024**3,
        nvme_bytes=8000 * 1024**3, nvme_seq_read_gbps=12.0, nvme_seq_write_gbps=10.0,
        pcie_bandwidth_gbps=64.0, gpu_interconnect_gbps=900.0,
        interconnect=InterconnectType.NVLINK, node_count=4,
    )

def cpu_only_topology() -> HardwareTopology:
    return HardwareTopology(
        gpus=(CPU_ONLY(),), cpu_cores=4, ram_bytes=8 * 1024**3,
        nvme_bytes=256 * 1024**3, nvme_seq_read_gbps=1.5, nvme_seq_write_gbps=1.0,
        pcie_bandwidth_gbps=8.0, gpu_interconnect_gbps=0.0,
        interconnect=InterconnectType.PCIE, node_count=1,
    )
