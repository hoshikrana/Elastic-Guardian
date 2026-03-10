"""Benchmark: All DSA structures under sustained load."""
import time
import unittest


class TestDSAThroughput(unittest.TestCase):
    def test_fibonacci_heap_10k_ops(self):
        from egx.intelligence.strategy.selector import FibonacciHeap
        h = FibonacciHeap()
        start = time.perf_counter()
        for i in range(10_000):
            h.insert(float(i), f"item_{i}")
        for _ in range(10_000):
            h.extract_max()
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 5.0, f"FibHeap 10K took {elapsed:.2f}s")

    def test_red_black_tree_10k_ops(self):
        from egx.intelligence.estimator.calibration.store import RedBlackTree
        t = RedBlackTree()
        start = time.perf_counter()
        for i in range(10_000):
            t.insert(i, f"v{i}")
        for i in range(10_000):
            t.search(i)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 5.0, f"RBTree 10K took {elapsed:.2f}s")

    def test_skip_list_10k_ops(self):
        from egx.orchestration.pressure.monitor import PressureEventSkipList
        sl = PressureEventSkipList()
        start = time.perf_counter()
        for i in range(10_000):
            sl.insert(float(i), f"event_{i}")
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 5.0, f"SkipList 10K took {elapsed:.2f}s")

    def test_segment_tree_10k_ops(self):
        from egx.intelligence.estimator.dryrun import MemorySegmentTree
        st = MemorySegmentTree(16384)
        start = time.perf_counter()
        for i in range(10_000):
            st.update(i % 16384, i * 100)
        for i in range(10_000):
            st.query_max(0, 8192)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 5.0, f"SegTree 10K took {elapsed:.2f}s")


if __name__ == "__main__":
    unittest.main()
