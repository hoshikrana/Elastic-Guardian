"""
EGX Test: core/memory/units.py
"""

import unittest
from egx.core.memory.units import to_bytes, from_bytes, GB, MB


class TestMemoryUnits(unittest.TestCase):
    def test_to_bytes(self):
        self.assertEqual(to_bytes(1, GB), 1024**3)
        self.assertEqual(to_bytes(512, MB), 512 * 1024**2)

    def test_from_bytes(self):
        self.assertEqual(from_bytes(1024**3, GB), 1.0)
        self.assertEqual(from_bytes(512 * 1024**2, MB), 512.0)


if __name__ == "__main__":
    unittest.main()
