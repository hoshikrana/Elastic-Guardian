"""
EGX Test: intelligence/estimator/calibration/regression.py
"""

import unittest
from egx.intelligence.estimator.calibration.regression import CalibrationRegression


class TestCalibrationRegression(unittest.TestCase):
    def test_initial_pass_through(self):
        reg = CalibrationRegression()
        self.assertEqual(reg.predict(1000), 1000)

    def test_calibration_updates(self):
        reg = CalibrationRegression()
        reg.update(100, 120)
        reg.update(200, 240)
        predicted = reg.predict(150)
        self.assertGreater(predicted, 0)


if __name__ == "__main__":
    unittest.main()
