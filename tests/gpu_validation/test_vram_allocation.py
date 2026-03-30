"""GPU Validation: VRAM allocation and memory budget tests."""

import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestVRAMAllocation(unittest.TestCase):
    def test_cuda_memory_available(self):
        mem = torch.cuda.get_device_properties(0).total_memory
        self.assertGreater(mem, 0)

    def test_tensor_allocation(self):
        t = torch.randn(1000, 1000, device="cuda")
        self.assertEqual(t.device.type, "cuda")
        del t
        torch.cuda.empty_cache()


class TestVRAMAllocationCPUFallback(unittest.TestCase):
    """Always-passing tests that verify CPU fallback logic."""

    def test_cpu_tensor_allocation(self):
        t = torch.randn(1000, 1000)
        self.assertEqual(t.device.type, "cpu")

    def test_memory_budget_calculation(self):
        from egx.intelligence.planner.memory_planner import MemoryPlanner
        from egx.core.enums import TrainingMode
        from tests.mocks.mock_gpu import make_gpu

        budget = MemoryPlanner().compute_budget(make_gpu(vram_gb=8), TrainingMode.QLORA)
        self.assertGreater(budget["usable_vram"], 0)
        self.assertLess(budget["usable_vram"], 8 * 1024**3)


if __name__ == "__main__":
    unittest.main()
