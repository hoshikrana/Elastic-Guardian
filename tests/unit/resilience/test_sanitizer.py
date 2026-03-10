"""Unit tests for InputSanitizer."""
import unittest
import torch


class TestInputSanitizer(unittest.TestCase):
    def test_clean_batch_passes(self):
        from egx.resilience.sanitizer import InputSanitizer
        san = InputSanitizer(strict=True)
        batch = {"x": torch.tensor([1.0, 2.0, 3.0])}
        clean = san.check_batch(batch)
        self.assertIn("x", clean)

    def test_nan_strict_raises(self):
        from egx.resilience.sanitizer import InputSanitizer
        san = InputSanitizer(strict=True)
        with self.assertRaises(ValueError):
            san.check_batch({"x": torch.tensor([float("nan")])})

    def test_inf_strict_raises(self):
        from egx.resilience.sanitizer import InputSanitizer
        san = InputSanitizer(strict=True)
        with self.assertRaises(ValueError):
            san.check_batch({"x": torch.tensor([float("inf")])})

    def test_nan_lenient_replaces(self):
        from egx.resilience.sanitizer import InputSanitizer
        san = InputSanitizer(strict=False)
        clean = san.check_batch({"x": torch.tensor([float("nan"), 1.0])})
        self.assertFalse(torch.isnan(clean["x"]).any())

    def test_check_loss_valid(self):
        from egx.resilience.sanitizer import InputSanitizer
        self.assertTrue(InputSanitizer().check_loss(torch.tensor(0.5)))

    def test_check_loss_nan(self):
        from egx.resilience.sanitizer import InputSanitizer
        self.assertFalse(InputSanitizer().check_loss(torch.tensor(float("nan"))))

    def test_stats_tracking(self):
        from egx.resilience.sanitizer import InputSanitizer
        san = InputSanitizer(strict=False)
        san.check_batch({"x": torch.tensor([float("nan")])})
        san.check_loss(torch.tensor(float("inf")))
        self.assertEqual(san.stats["nan_count"], 1)
        self.assertEqual(san.stats["inf_count"], 1)

    def test_integer_tensor_passthrough(self):
        from egx.resilience.sanitizer import InputSanitizer
        san = InputSanitizer(strict=True)
        batch = {"ids": torch.tensor([1, 2, 3])}
        clean = san.check_batch(batch)
        self.assertTrue(torch.equal(clean["ids"], batch["ids"]))


if __name__ == "__main__":
    unittest.main()
