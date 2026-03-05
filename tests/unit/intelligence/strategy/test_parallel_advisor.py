"""
EGX Test: intelligence/strategy/parallel_advisor.py
"""
import unittest
from egx.intelligence.strategy.parallel_advisor import ParallelAdvisor
from egx.core.models import HardwareTopology
from egx.core.enums import InterconnectType

class TestParallelAdvisor(unittest.TestCase):
    def test_single_device(self):
        topo = HardwareTopology(
            gpus=(), cpu_cores=8, ram_bytes=32*1024**3,
            nvme_bytes=500*1024**3, nvme_seq_read_gbps=3.5,
            nvme_seq_write_gbps=2.5, pcie_bandwidth_gbps=15.8,
            gpu_interconnect_gbps=15.8, interconnect=InterconnectType.PCIE,
            node_count=1
        )
        config = ParallelAdvisor().advise(topo, model_params=7_000_000_000)
        self.assertEqual(config.data_parallel_size, 1)

if __name__ == "__main__":
    unittest.main()
