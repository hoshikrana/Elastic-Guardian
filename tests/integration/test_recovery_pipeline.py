"""Integration: NaN/Inf injection → sanitizer catches → validates loss."""
import unittest
import torch


class TestRecoveryPipeline(unittest.TestCase):
    def test_nan_batch_strict_raises(self):
        from egx.resilience.sanitizer import InputSanitizer
        san = InputSanitizer(strict=True)
        batch = {"x": torch.tensor([1.0, float("nan"), 3.0])}
        with self.assertRaises(ValueError):
            san.check_batch(batch)

    def test_inf_batch_strict_raises(self):
        from egx.resilience.sanitizer import InputSanitizer
        san = InputSanitizer(strict=True)
        batch = {"x": torch.tensor([1.0, float("inf"), 3.0])}
        with self.assertRaises(ValueError):
            san.check_batch(batch)

    def test_nan_batch_lenient_replaces(self):
        from egx.resilience.sanitizer import InputSanitizer
        san = InputSanitizer(strict=False)
        batch = {"x": torch.tensor([1.0, float("nan"), 3.0])}
        clean = san.check_batch(batch)
        self.assertFalse(torch.isnan(clean["x"]).any())

    def test_loss_check_valid(self):
        from egx.resilience.sanitizer import InputSanitizer
        san = InputSanitizer()
        self.assertTrue(san.check_loss(torch.tensor(0.5)))

    def test_loss_check_nan(self):
        from egx.resilience.sanitizer import InputSanitizer
        san = InputSanitizer()
        self.assertFalse(san.check_loss(torch.tensor(float("nan"))))

    def test_loss_check_inf(self):
        from egx.resilience.sanitizer import InputSanitizer
        san = InputSanitizer()
        self.assertFalse(san.check_loss(torch.tensor(float("inf"))))

    def test_sanitizer_stats(self):
        from egx.resilience.sanitizer import InputSanitizer
        san = InputSanitizer(strict=False)
        san.check_batch({"x": torch.tensor([float("nan")])})
        san.check_loss(torch.tensor(float("inf")))
        stats = san.stats
        self.assertEqual(stats["nan_count"], 1)
        self.assertEqual(stats["inf_count"], 1)


if __name__ == "__main__":
    unittest.main()
