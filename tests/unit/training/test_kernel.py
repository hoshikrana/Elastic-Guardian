"""
EGX Test: training/kernel.py
"""
import unittest
import torch
import torch.nn as nn
from egx.training.kernel import TrainingKernel

class TestTrainingKernel(unittest.TestCase):
    def test_train_step(self):
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 1)
            def forward(self, input, target):
                class Out: pass
                o = Out()
                o.loss = (self.layer(input) - target).pow(2).mean()
                return o

        model = MockModel()
        kernel = TrainingKernel(model, learning_rate=0.01)
        batch = {"input": torch.randn(2, 10), "target": torch.randn(2, 1)}
        
        loss = kernel.train_step(batch, step=1)
        self.assertGreater(loss, 0)

if __name__ == "__main__":
    unittest.main()
