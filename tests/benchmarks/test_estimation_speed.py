"""Benchmark: Estimator performance regression tests."""
import time
import unittest
from tests.mocks.mock_gpu import make_gpu


class TestEstimationSpeed(unittest.TestCase):
    def test_hybrid_estimator_under_100ms(self):
        from egx.intelligence.estimator.hybrid import HybridEstimator
        gpu = make_gpu(vram_gb=80)
        est = HybridEstimator()
        start = time.perf_counter()
        result = est.estimate_vram(
            model_params=7_000_000_000,
            bytes_per_param=2,
            gpu=gpu,
            mode="qlora",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertGreater(result, 0)
        self.assertLess(elapsed_ms, 100, f"Hybrid took {elapsed_ms:.2f}ms, expected < 100ms")

    def test_calibration_cache_speed(self):
        from egx.intelligence.estimator.calibration.cache import CalibrationCache
        cache = CalibrationCache(max_size=1000)
        start = time.perf_counter()
        for i in range(1000):
            cache.put(f"key_{i}", {"vram_bytes": i * 1000})
        for i in range(1000):
            cache.get(f"key_{i}")
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertLess(elapsed_ms, 100, f"Cache 1K ops took {elapsed_ms:.2f}ms")

    def test_calibration_regression_speed(self):
        from egx.intelligence.estimator.calibration.regression import CalibrationRegression
        reg = CalibrationRegression()
        start = time.perf_counter()
        for i in range(100):
            reg.update(i * 1000, i * 1200)
        for i in range(1000):
            reg.predict(i * 500)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertLess(elapsed_ms, 100, f"Regression took {elapsed_ms:.2f}ms")


if __name__ == "__main__":
    unittest.main()
