"""
EGX Test: intelligence/strategy/scorer.py
"""

import unittest
from egx.intelligence.strategy.scorer import StrategyScorer
from egx.core.models import GPUSpec
from egx.core.enums import TrainingMode


class TestStrategyScorer(unittest.TestCase):
    def test_scoring(self):
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
        results = StrategyScorer().score_all(
            gpu,
            model_bytes=4 * 1024**3,
            modes=[TrainingMode.FULL_FINETUNE, TrainingMode.LORA, TrainingMode.QLORA],
        )
        self.assertEqual(len(results), 3)
        self.assertTrue(results[0].score >= results[-1].score)


if __name__ == "__main__":
    unittest.main()
