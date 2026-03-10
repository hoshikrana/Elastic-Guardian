"""Unit tests for RAMToNVMeSwapper."""
import unittest
import torch


class TestRAMToNVMeSwapper(unittest.TestCase):
    def test_offload_and_restore(self):
        from egx.orchestration.swapper.ram_to_nvme import RAMToNVMeSwapper
        sw = RAMToNVMeSwapper()
        t = torch.randn(50)
        sw.offload("test_t", t)
        self.assertEqual(sw.cached_count, 1)
        restored = sw.restore("test_t")
        self.assertEqual(restored.shape, t.shape)
        sw.cleanup()

    def test_cleanup(self):
        from egx.orchestration.swapper.ram_to_nvme import RAMToNVMeSwapper
        sw = RAMToNVMeSwapper()
        sw.offload("a", torch.randn(10))
        sw.offload("b", torch.randn(10))
        self.assertEqual(sw.cached_count, 2)
        sw.cleanup()
        self.assertEqual(sw.cached_count, 0)

    def test_restore_missing_key(self):
        from egx.orchestration.swapper.ram_to_nvme import RAMToNVMeSwapper
        sw = RAMToNVMeSwapper()
        with self.assertRaises(KeyError):
            sw.restore("nonexistent")
        sw.cleanup()


if __name__ == "__main__":
    unittest.main()
