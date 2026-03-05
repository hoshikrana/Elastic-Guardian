"""
EGX Topology Builder — Layer 2.

Assembles HardwareTopology from probes.
"""

from __future__ import annotations

import psutil
from typing import List
from egx.core.models import HardwareTopology, GPUSpec
from egx.core.enums import InterconnectType


class TopologyBuilder:
    """
    Law 1: Assembles the topology contract.
    """
    
    def build(self, gpus: List[GPUSpec]) -> HardwareTopology:
        # CPU Info
        cpu_cores = psutil.cpu_count(logical=False) or 1
        ram_bytes = psutil.virtual_memory().total
        
        # Interconnect Heuristic (Default to PCIE, updated by sampler)
        interconnect = InterconnectType.PCIE
        if any(g.nvlink_peer_ids for g in gpus):
            interconnect = InterconnectType.NVLINK
            
        return HardwareTopology(
            gpus=tuple(gpus),
            cpu_cores=cpu_cores,
            ram_bytes=ram_bytes,
            nvme_bytes=100 * 1024 * 1024 * 1024, # Default 100GB, updated by probe
            nvme_seq_read_gbps=3.5,
            nvme_seq_write_gbps=2.5,
            pcie_bandwidth_gbps=15.8, # PCIe 3.0 x16 approx
            gpu_interconnect_gbps=600.0 if interconnect == InterconnectType.NVLINK else 15.8,
            interconnect=interconnect,
            node_count=1
        )