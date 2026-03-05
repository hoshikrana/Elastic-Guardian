"""
EGX Test: peft/lora.py
"""
import unittest
import torch
import torch.nn as nn
from egx.peft.lora import LoRALinear

class TestLoRA(unittest.TestCase):
    def test_lora_linear(self):
        orig = nn.Linear(64, 32)
        lora = LoRALinear(orig, rank=4, alpha=8)
        x = torch.randn(2, 64)
        out = lora(x)
        self.assertEqual(out.shape, (2, 32))

    def test_trainable_params(self):
        orig = nn.Linear(64, 32)
        lora = LoRALinear(orig, rank=4)
        self.assertEqual(lora.trainable_params, 4*64 + 32*4)

if __name__ == "__main__":
    unittest.main()
