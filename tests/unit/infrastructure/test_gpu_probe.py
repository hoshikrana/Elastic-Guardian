"""
EGX Test: infrastructure/gpu_probe.py
"""
import unittest
from egx.infrastructure.gpu_probe import GPUProber

class TestGPUProber(unittest.TestCase):
    def test_probe_returns_list(self):
        prober = GPUProber()
        gpus = prober.probe()
        self.assertIsInstance(gpus, list)
        self.assertGreater(len(gpus), 0)

    def test_cpu_fallback(self):
        prober = GPUProber()
        # Mocking no GPU available
        with unittest.mock.patch('torch.cuda.is_available', return_value=False):
            with unittest.mock.patch('torch.backends.mps.is_available', return_value=False):
                with unittest.mock.patch('pynvml.nvmlInit', side_effect=Exception()):
                    gpus = prober.probe()
                    self.assertEqual(gpus[0].vendor, "cpu")

if __name__ == "__main__":
    unittest.main()
