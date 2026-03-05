"""
EGX Test: intelligence/strategy/selector.py (DSA-1: Fibonacci Heap)
"""
import unittest
from egx.intelligence.strategy.selector import FibonacciHeap

class TestFibonacciHeap(unittest.TestCase):
    def test_insert_extract(self):
        h = FibonacciHeap()
        h.insert(10.0, "a")
        h.insert(50.0, "b")
        h.insert(30.0, "c")
        self.assertEqual(h.extract_max().value, "b")
        self.assertEqual(h.extract_max().value, "c")

    def test_increase_key(self):
        h = FibonacciHeap()
        h.insert(10.0, "low")
        node = h.insert(20.0, "mid")
        h.insert(50.0, "high")
        h.increase_key(node, 60.0)
        self.assertEqual(h.extract_max().value, "mid")

if __name__ == "__main__":
    unittest.main()
