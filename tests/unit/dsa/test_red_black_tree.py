"""
EGX Test: intelligence/estimator/calibration/store.py (DSA-2: Red-Black Tree)
"""

import unittest
from egx.intelligence.estimator.calibration.store import RedBlackTree


class TestRedBlackTree(unittest.TestCase):
    def test_insert_search(self):
        t = RedBlackTree()
        t.insert(10, "v1")
        t.insert(20, "v2")
        self.assertEqual(t.search(10), "v1")
        self.assertEqual(t.search(20), "v2")
        self.assertIsNone(t.search(99))

    def test_find_nearest(self):
        t = RedBlackTree()
        t.insert(10, "a")
        t.insert(20, "b")
        t.insert(30, "c")
        key, val = t.find_nearest(22)
        self.assertEqual(val, "b")


if __name__ == "__main__":
    unittest.main()
