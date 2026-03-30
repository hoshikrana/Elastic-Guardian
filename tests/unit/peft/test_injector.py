"""Unit tests for PEFT injection (LoRA injection utility)."""

import unittest
import torch
import torch.nn as nn


class TestPEFTInjector(unittest.TestCase):
    def test_lora_injection(self):
        from egx.peft.lora import inject_lora, LoRALinear

        model = nn.Module()
        model.q_proj = nn.Linear(64, 64)
        model.v_proj = nn.Linear(64, 64)
        model = inject_lora(model, rank=4, targets=["q_proj", "v_proj"])
        self.assertIsInstance(model.q_proj, LoRALinear)
        self.assertIsInstance(model.v_proj, LoRALinear)

    def test_lora_forward_preserves_shape(self):
        from egx.peft.lora import LoRALinear

        orig = nn.Linear(32, 16)
        lora = LoRALinear(orig, rank=4)
        x = torch.randn(2, 32)
        out = lora(x)
        self.assertEqual(out.shape, (2, 16))

    def test_lora_trainable_params(self):
        from egx.peft.lora import LoRALinear

        lora = LoRALinear(nn.Linear(64, 32), rank=8)
        self.assertEqual(lora.trainable_params, 8 * 64 + 32 * 8)

    def test_qlora_forward(self):
        from egx.peft.qlora import QuantizedLinear

        q = QuantizedLinear(nn.Linear(64, 32), rank=4)
        out = q(torch.randn(2, 64))
        self.assertEqual(out.shape, (2, 32))


if __name__ == "__main__":
    unittest.main()
