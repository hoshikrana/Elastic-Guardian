"""
EGX Test: orchestration/pressure/monitor.py (DSA-3: Skip List)
"""

import unittest
from egx.orchestration.pressure.monitor import PressureEventSkipList


class TestSkipList(unittest.TestCase):
    def test_insert_latest(self):
        sl = PressureEventSkipList()
        sl.insert(1.0, "e1")
        sl.insert(3.0, "e3")
        sl.insert(2.0, "e2")
        self.assertEqual(sl.latest(), "e3")


if __name__ == "__main__":
    unittest.main()
