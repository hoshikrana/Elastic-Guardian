"""Unit tests for HybridEstimator."""

import unittest
from tests.mocks.mock_gpu import make_gpu


class TestHybridEstimator(unittest.TestCase):
    def test_estimate_vram(self):
        from egx.intelligence.estimator.hybrid import HybridEstimator

        est = HybridEstimator()
        result = est.estimate_vram(
            model_params=1_000_000, bytes_per_param=2, gpu=make_gpu(), mode="lora"
        )
        self.assertGreater(result, 0)

    def test_caching(self):
        from egx.intelligence.estimator.hybrid import HybridEstimator

        est = HybridEstimator()
        gpu = make_gpu(name="CacheTestGPU")
        r1 = est.estimate_vram(
            model_params=500, bytes_per_param=4, gpu=gpu, mode="full"
        )
        r2 = est.estimate_vram(
            model_params=500, bytes_per_param=4, gpu=gpu, mode="full"
        )
        self.assertEqual(r1, r2)

    def test_record_actual_updates_regression(self):
        from egx.intelligence.estimator.hybrid import HybridEstimator

        est = HybridEstimator()
        est.record_actual(predicted=1000, actual=1200)
        est.record_actual(predicted=2000, actual=2400)
        self.assertIsNotNone(est._regression)


if __name__ == "__main__":
    unittest.main()
