"""
EGX Test: intelligence/planner/memory_planner.py
"""

import unittest
from egx.intelligence.planner.memory_planner import MemoryPlanner
from egx.core.models import GPUSpec
from egx.core.enums import TrainingMode


class TestMemoryPlanner(unittest.TestCase):
    def test_budget_computation(self):
        gpu = GPUSpec(
            device_id=0,
            name="T",
            vram_bytes=8 * 1024**3,
            compute_capability=(8, 0),
            memory_bandwidth_gbps=400.0,
            fp16_tflops=20.0,
            bf16_tflops=20.0,
            supports_flash_attn2=True,
            supports_fp8=False,
            nvlink_peer_ids=(),
        )
        budget = MemoryPlanner().compute_budget(gpu, TrainingMode.QLORA)
        self.assertGreater(budget["usable_vram"], 0)
        self.assertLess(budget["usable_vram"], budget["total_vram"])


if __name__ == "__main__":
    unittest.main()
