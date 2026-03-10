"""GPU Validation: OOM recovery and elastic batch integration."""
import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestOOMRecoveryGPU(unittest.TestCase):
    def test_empty_cache_on_oom(self):
        from egx.orchestration.pressure.elastic_batch import ElasticBatchResizer
        eb = ElasticBatchResizer(initial_batch=32)
        new = eb.on_oom()
        self.assertEqual(new, 16)


class TestOOMRecoveryCPU(unittest.TestCase):
    """OOM recovery logic tests that work on CPU."""
    def test_oom_halves_batch(self):
        from egx.orchestration.pressure.elastic_batch import ElasticBatchResizer
        eb = ElasticBatchResizer(initial_batch=64)
        self.assertEqual(eb.on_oom(), 32)
        self.assertEqual(eb.on_oom(), 16)

    def test_oom_floor_respected(self):
        from egx.orchestration.pressure.elastic_batch import ElasticBatchResizer
        eb = ElasticBatchResizer(initial_batch=2, min_batch=1)
        eb.on_oom()
        self.assertEqual(eb.on_oom(), 1)


if __name__ == "__main__":
    unittest.main()
