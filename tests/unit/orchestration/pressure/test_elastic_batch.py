"""
EGX Test: orchestration/pressure/elastic_batch.py
"""
import unittest
from egx.orchestration.pressure.elastic_batch import ElasticBatchResizer

class TestElasticBatch(unittest.TestCase):
    def test_oom_halves(self):
        eb = ElasticBatchResizer(initial_batch=32)
        new_batch = eb.on_oom()
        self.assertEqual(new_batch, 16)
        new_batch = eb.on_oom()
        self.assertEqual(new_batch, 8)

    def test_min_batch(self):
        eb = ElasticBatchResizer(initial_batch=2, min_batch=1)
        eb.on_oom()
        result = eb.on_oom()
        self.assertEqual(result, 1)

if __name__ == "__main__":
    unittest.main()
