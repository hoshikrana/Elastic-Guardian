"""
EGX Test: core/models.py
"""

import unittest
from egx.core.models import GPUSpec, HardwareTopology
from egx.core.enums import HardwareTier, InterconnectType


class TestCoreModels(unittest.TestCase):
    def test_gpu_spec_frozen(self):
        spec = GPUSpec(
            device_id=0,
            name="Test",
            vram_bytes=8 * 1024**3,
            compute_capability=(8, 0),
            memory_bandwidth_gbps=400.0,
            fp16_tflops=20.0,
            bf16_tflops=20.0,
            supports_flash_attn2=True,
            supports_fp8=False,
            nvlink_peer_ids=(),
        )
        with self.assertRaises(AttributeError):
            spec.vram_bytes = 0  # type: ignore

    def test_gpu_spec_tier(self):
        spec = GPUSpec(
            device_id=0,
            name="T",
            vram_bytes=80 * 1024**3,
            compute_capability=(9, 0),
            memory_bandwidth_gbps=3350.0,
            fp16_tflops=989.0,
            bf16_tflops=989.0,
            supports_flash_attn2=True,
            supports_fp8=True,
            nvlink_peer_ids=(),
        )
        self.assertEqual(spec.tier, HardwareTier.DATACENTER)

    def test_hardware_topology_frozen(self):
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
        with self.assertRaises(AttributeError):
            topo.cpu_cores = 16  # type: ignore


if __name__ == "__main__":
    unittest.main()
