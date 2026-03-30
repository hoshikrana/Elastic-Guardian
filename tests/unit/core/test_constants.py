"""
EGX Test: core/constants.py
"""

import unittest
from egx.core.constants import KB, MB, GB, TB, SAFETY_THRESHOLDS
from egx.core.enums import TrainingMode


class TestCoreConstants(unittest.TestCase):
    def test_units(self):
        self.assertEqual(KB, 1024)
        self.assertEqual(MB, 1024 * 1024)
        self.assertEqual(GB, 1024**3)
        self.assertEqual(TB, 1024**4)

    def test_safety_thresholds(self):
        self.assertIn(TrainingMode.QLORA, SAFETY_THRESHOLDS)
        self.assertTrue(0 < SAFETY_THRESHOLDS[TrainingMode.QLORA] <= 1.0)


if __name__ == "__main__":
    unittest.main()
