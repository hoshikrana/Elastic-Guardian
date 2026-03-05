"""
EGX Test: api/trainer.py
"""
import unittest
from egx.api.trainer import EGX

class TestEGXTrainer(unittest.TestCase):
    def test_init_no_args(self):
        trainer = EGX()
        self.assertIsNotNone(trainer)

if __name__ == "__main__":
    unittest.main()
