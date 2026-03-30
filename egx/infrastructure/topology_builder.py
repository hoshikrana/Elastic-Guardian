"""
EGX Topology Builder — Layer 2.

Assembles HardwareTopology from probes.
"""

from __future__ import annotations

import psutil
from typing import List
from egx.core.models import HardwareTopology, GPUSpec
from egx.core.enums import InterconnectType
from egx.core.interfaces import BaseTopologyBuilder


class TopologyBuilder(BaseTopologyBuilder):
    """
    Law 1: Assembles the topology contract.
    """

    def build(self, gpus: List[GPUSpec]) -> HardwareTopology:
        # CPU Info
        cpu_cores = psutil.cpu_count(logical=False) or 1
        ram_bytes = psutil.virtual_memory().total

        # NVMe Probe
        try:
            from egx.infrastructure.nvme_probe import NVMeProber

            with NVMeProber() as prober:
                nvme_read, nvme_write, nvme_capacity = prober.probe()
        except Exception as e:
            nvme_read, nvme_write, nvme_capacity = 3.5, 2.5, 100 * 1024 * 1024 * 1024

        # PCIe Bandwidth Sample
        pcie_gbps = 15.8
        try:
            if gpus:
                import torch

                if torch.cuda.is_available():
                    from egx.infrastructure.bandwidth_sampler import BandwidthSampler

                    pcie_gbps = BandwidthSampler().sample_pcie(gpus[0].device_id)
        except Exception:
            pass

        # Interconnect Heuristic (Default to PCIE, updated by sampler)
        interconnect = InterconnectType.PCIE
        if any(g.nvlink_peer_ids for g in gpus):
            interconnect = InterconnectType.NVLINK

        return HardwareTopology(
            gpus=tuple(gpus),
            cpu_cores=cpu_cores,
            ram_bytes=ram_bytes,
            nvme_bytes=nvme_capacity,
            nvme_seq_read_gbps=nvme_read,
            nvme_seq_write_gbps=nvme_write,
            pcie_bandwidth_gbps=pcie_gbps,
            gpu_interconnect_gbps=(
                600.0 if interconnect == InterconnectType.NVLINK else pcie_gbps
            ),
            interconnect=interconnect,
            node_count=1,
        )
