"""Unit tests for GPUDataPrefetcher."""
import unittest
import torch


class TestGPUDataPrefetcher(unittest.TestCase):
    def test_cpu_prefetch(self):
        from egx.data.prefetcher import GPUDataPrefetcher
        batches = [{"x": torch.randn(2, 4)}, {"x": torch.randn(2, 4)}]
        pf = GPUDataPrefetcher(iter(batches), device=torch.device("cpu"))
        collected = []
        for batch in pf:
            collected.append(batch)
        self.assertEqual(len(collected), 2)

    def test_empty_loader(self):
        from egx.data.prefetcher import GPUDataPrefetcher
        pf = GPUDataPrefetcher(iter([]), device=torch.device("cpu"))
        with self.assertRaises(StopIteration):
            next(pf)


if __name__ == "__main__":
    unittest.main()
