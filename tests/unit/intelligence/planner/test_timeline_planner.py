"""
EGX Test: intelligence/planner/timeline_planner.py
"""
import unittest
from egx.intelligence.planner.timeline_planner import TimelinePlanner
from egx.core.models import GPUSpec

class TestTimelinePlanner(unittest.TestCase):
    def test_estimate(self):
        gpu = GPUSpec(
            device_id=0, name="T", vram_bytes=8*1024**3,
            compute_capability=(8,0), memory_bandwidth_gbps=400.0,
            fp16_tflops=20.0, bf16_tflops=20.0,
            supports_flash_attn2=True, supports_fp8=False, nvlink_peer_ids=()
        )
        result = TimelinePlanner().estimate(
            total_tokens=1_000_000, seq_length=512, batch_size=4, gpu=gpu
        )
        self.assertGreater(result.total_steps, 0)
        self.assertGreater(result.estimated_hours, 0)

if __name__ == "__main__":
    unittest.main()
