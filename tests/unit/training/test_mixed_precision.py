"""
EGX Test: training/mixed_precision.py
"""

import unittest
from egx.training.mixed_precision import PrecisionSelector
from egx.core.models import GPUSpec, HardwareTopology
from egx.core.enums import DType, InterconnectType


class TestMixedPrecision(unittest.TestCase):
    def test_select_optimal(self):
        gpu = GPUSpec(
            device_id=0,
            name="A100",
            vram_bytes=40 * 1024**3,
            compute_capability=(8, 0),
            memory_bandwidth_gbps=1555.0,
            fp16_tflops=312.0,
            bf16_tflops=312.0,
            supports_flash_attn2=True,
            supports_fp8=False,
            nvlink_peer_ids=(),
        )
        topo = HardwareTopology(
            gpus=(gpu,),
            cpu_cores=64,
            ram_bytes=256 * 1024**3,
            nvme_bytes=2000 * 1024**3,
            nvme_seq_read_gbps=7.0,
            nvme_seq_write_gbps=5.0,
            pcie_bandwidth_gbps=31.5,
            gpu_interconnect_gbps=600.0,
            interconnect=InterconnectType.NVLINK,
            node_count=1,
        )
        dtype, autocast = PrecisionSelector.select_optimal(topo)
        self.assertEqual(dtype, DType.BF16)
        self.assertTrue(autocast)


if __name__ == "__main__":
    unittest.main()
