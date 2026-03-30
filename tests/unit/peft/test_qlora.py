"""
EGX Test: peft/qlora.py
"""

import unittest
import torch
import torch.nn as nn
from egx.peft.qlora import QuantizedLinear


class TestQLoRA(unittest.TestCase):
    def test_quantized_linear(self):
        orig = nn.Linear(64, 32)
        q = QuantizedLinear(orig, rank=4)
        x = torch.randn(2, 64)
        out = q(x)
        self.assertEqual(out.shape, (2, 32))


if __name__ == "__main__":
    unittest.main()
