"""
EGX Test: intelligence/estimator/dryrun.py (DSA-4: Segment Tree)
"""

import unittest
from egx.intelligence.estimator.dryrun import MemorySegmentTree


class TestSegmentTree(unittest.TestCase):
    def test_range_max(self):
        st = MemorySegmentTree(8)
        st.update(0, 100)
        st.update(3, 500)
        st.update(5, 200)
        self.assertEqual(st.query_max(0, 4), 500)
        self.assertEqual(st.global_peak(), 500)


if __name__ == "__main__":
    unittest.main()
