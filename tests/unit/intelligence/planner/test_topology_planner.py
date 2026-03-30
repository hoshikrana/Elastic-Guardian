"""
EGX Test: intelligence/planner/topology_planner.py
"""

import unittest
from egx.intelligence.planner.topology_planner import TopologyPlanner
from egx.core.models import HardwareTopology
from egx.core.enums import InterconnectType


class TestTopologyPlanner(unittest.TestCase):
    def test_single_gpu(self):
        topo = HardwareTopology(
            gpus=(),
            cpu_cores=8,
            ram_bytes=32 * 1024**3,
            nvme_bytes=500 * 1024**3,
            nvme_seq_read_gbps=3.5,
            nvme_seq_write_gbps=2.5,
            pcie_bandwidth_gbps=15.8,
            gpu_interconnect_gbps=15.8,
            interconnect=InterconnectType.PCIE,
            node_count=1,
        )
        plan = TopologyPlanner().plan(topo, model_bytes=1 * 1024**3)
        self.assertEqual(plan.strategy, "single")


if __name__ == "__main__":
    unittest.main()
