"""
EGX Test: core/memory/value.py
"""
import unittest
from egx.core.memory.value import MemoryValue
from egx.core.exceptions import NegativeMemoryError

class TestMemoryValue(unittest.TestCase):
    def test_creation(self):
        mv = MemoryValue(1024)
        self.assertEqual(mv.bytes, 1024)

    def test_bool_rejected(self):
        # Law 10: bools are not allowed as memory ints
        from egx.core.exceptions import BoolAsIntError
        with self.assertRaises(BoolAsIntError):
            MemoryValue(True)

    def test_negative_rejected(self):
        with self.assertRaises(NegativeMemoryError):
            MemoryValue(-1)

    def test_immutable(self):
        mv = MemoryValue(1024)
        with self.assertRaises(AttributeError):
            mv.bytes = 2048  # type: ignore

if __name__ == "__main__":
    unittest.main()
