"""Unit tests for LRUEvictionPolicy."""

import unittest


class TestLRUEvictionPolicy(unittest.TestCase):
    def test_access_and_evict(self):
        from egx.orchestration.pressure.eviction_policy import LRUEvictionPolicy

        pol = LRUEvictionPolicy(capacity_bytes=1000)
        pol.access("t1", 400)
        pol.access("t2", 400)
        evicted = pol.evict_until(300)
        self.assertIn("t1", evicted)

    def test_usage_pct(self):
        from egx.orchestration.pressure.eviction_policy import LRUEvictionPolicy

        pol = LRUEvictionPolicy(capacity_bytes=1000)
        pol.access("t1", 500)
        self.assertAlmostEqual(pol.usage_pct, 0.5)

    def test_free_bytes(self):
        from egx.orchestration.pressure.eviction_policy import LRUEvictionPolicy

        pol = LRUEvictionPolicy(capacity_bytes=1000)
        pol.access("t1", 300)
        self.assertEqual(pol.free_bytes, 700)

    def test_move_to_end_on_re_access(self):
        from egx.orchestration.pressure.eviction_policy import LRUEvictionPolicy

        pol = LRUEvictionPolicy(capacity_bytes=1000)
        pol.access("t1", 300)
        pol.access("t2", 300)
        pol.access("t1", 300)  # re-access t1, should move to end
        evicted = pol.evict_until(500)
        self.assertIn("t2", evicted)
        self.assertNotIn("t1", evicted)


if __name__ == "__main__":
    unittest.main()
