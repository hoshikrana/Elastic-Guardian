"""
EGX Test: intelligence/estimator/calibration/cache.py
"""

import unittest
from egx.intelligence.estimator.calibration.cache import CalibrationCache


class TestCalibrationCache(unittest.TestCase):
    def test_put_get(self):
        cache = CalibrationCache(max_size=10)
        cache.put("k1", {"vram_bytes": 100})
        self.assertEqual(cache.get("k1")["vram_bytes"], 100)

    def test_miss(self):
        cache = CalibrationCache()
        self.assertIsNone(cache.get("nonexistent"))

    def test_lru_eviction(self):
        cache = CalibrationCache(max_size=2)
        cache.put("a", {"v": 1})
        cache.put("b", {"v": 2})
        cache.put("c", {"v": 3})
        self.assertIsNone(cache.get("a"))
        self.assertIsNotNone(cache.get("b"))


if __name__ == "__main__":
    unittest.main()
