"""Unit tests for AnalyticalEstimator."""
import unittest


class TestAnalyticalEstimator(unittest.TestCase):
    def test_instantiation(self):
        from egx.intelligence.estimator.analytical import AnalyticalEstimator
        est = AnalyticalEstimator()
        self.assertIsNotNone(est)
        self.assertEqual(est.ACTIVATION_FACTOR_DEFAULT, 34.0)

    def test_inherits_base(self):
        from egx.intelligence.estimator.analytical import AnalyticalEstimator
        from egx.intelligence.estimator.base import BaseEstimator
        self.assertTrue(issubclass(AnalyticalEstimator, BaseEstimator))


if __name__ == "__main__":
    unittest.main()
