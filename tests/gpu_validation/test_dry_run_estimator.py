"""GPU Validation: Dry-run memory estimator."""
import unittest


class TestDryRunEstimator(unittest.TestCase):
    def test_segment_tree_peak_tracking(self):
        from egx.intelligence.estimator.dryrun import MemorySegmentTree
        st = MemorySegmentTree(1024)
        st.update(0, 1000)
        st.update(100, 5000)
        st.update(500, 3000)
        self.assertEqual(st.global_peak(), 5000)

    def test_segment_tree_range_query(self):
        from egx.intelligence.estimator.dryrun import MemorySegmentTree
        st = MemorySegmentTree(8)
        st.update(0, 100)
        st.update(3, 500)
        st.update(5, 200)
        self.assertEqual(st.query_max(0, 4), 500)
        self.assertEqual(st.query_max(4, 6), 200)

    def test_segment_tree_empty(self):
        from egx.intelligence.estimator.dryrun import MemorySegmentTree
        st = MemorySegmentTree(16)
        self.assertEqual(st.global_peak(), 0)


if __name__ == "__main__":
    unittest.main()
