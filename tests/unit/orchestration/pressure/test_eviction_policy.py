"""
EGX Test: orchestration/pressure/eviction_policy.py
"""
import unittest
from egx.orchestration.pressure.eviction_policy import LRUEvictionPolicy

class TestEvictionPolicy(unittest.TestCase):
    def test_eviction(self):
        pol = LRUEvictionPolicy(capacity_bytes=1000)
        pol.access("t1", 400)
        pol.access("t2", 400)
        evicted = pol.evict_until(300)
        self.assertIn("t1", evicted)

if __name__ == "__main__":
    unittest.main()
