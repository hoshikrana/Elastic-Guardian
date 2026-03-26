"""
EGX Test: runtime/engine.py
"""
import unittest
import torch
from egx.api.trainer import EGX

class TestEGXEngine(unittest.TestCase):
    def test_boot_sequence(self):
        egx = EGX()
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.randn(1))
            def forward(self, **kwargs):
                class Out: 
                    def __init__(self, p): self.loss = p.sum()
                return Out(self.p)
                
        result = egx.train(model=MockModel(), dataset=[])
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["topology"])

if __name__ == "__main__":
    unittest.main()
