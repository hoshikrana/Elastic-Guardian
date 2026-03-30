"""
EGX Test: intelligence/strategy/batch_optimizer.py (DSA-7: Binary Search)
"""

import unittest
from egx.intelligence.strategy.batch_optimizer import find_max_batch_size


class TestBinarySearch(unittest.TestCase):
    def test_find_max(self):
        result = find_max_batch_size(lambda x: x <= 100, low=1, high=200)
        self.assertEqual(result, 100)


if __name__ == "__main__":
    unittest.main()
