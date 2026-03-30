"""
EGX Test: orchestration/swapper/ram_to_nvme.py
"""

import unittest
import torch
from egx.orchestration.swapper.ram_to_nvme import RAMToNVMeSwapper


class TestRAMToNVMeSwapper(unittest.TestCase):
    def test_offload_restore(self):
        sw = RAMToNVMeSwapper()
        t = torch.randn(100)
        sw.offload("test_tensor", t)
        self.assertEqual(sw.cached_count, 1)
        restored = sw.restore("test_tensor")
        self.assertEqual(restored.shape, t.shape)
        sw.cleanup()
        self.assertEqual(sw.cached_count, 0)


if __name__ == "__main__":
    unittest.main()
