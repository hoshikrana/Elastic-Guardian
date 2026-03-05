"""
EGX Test: resilience/sanitizer.py
"""
import unittest
import torch
from egx.resilience.sanitizer import InputSanitizer

class TestInputSanitizer(unittest.TestCase):
    def test_clean_batch(self):
        san = InputSanitizer(strict=True)
        batch = {"input_ids": torch.tensor([1, 2, 3])}
        clean = san.check_batch(batch)
        self.assertIn("input_ids", clean)

    def test_nan_detection_strict(self):
        san = InputSanitizer(strict=True)
        batch = {"x": torch.tensor([1.0, float('nan'), 3.0])}
        with self.assertRaises(ValueError):
            san.check_batch(batch)

    def test_loss_check(self):
        san = InputSanitizer()
        self.assertTrue(san.check_loss(torch.tensor(0.5)))
        self.assertFalse(san.check_loss(torch.tensor(float('nan'))))

if __name__ == "__main__":
    unittest.main()
