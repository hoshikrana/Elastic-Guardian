"""Integration: LoRA injection → forward pass → param count."""

import unittest
import torch
import torch.nn as nn


class TestPEFTPipeline(unittest.TestCase):
    def test_lora_injection_forward(self):
        from egx.peft.lora import inject_lora, LoRALinear
        from tests.mocks.mock_model import ModelWithQProj

        model = ModelWithQProj()
        original_params = sum(p.numel() for p in model.parameters())
        model = inject_lora(model, rank=4, targets=["q_proj"])
        self.assertIsInstance(model.q_proj, LoRALinear)
        x = torch.randn(2, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 64))

    def test_lora_reduces_trainable_params(self):
        from egx.peft.lora import inject_lora
        from tests.mocks.mock_model import ModelWithQProj

        model = ModelWithQProj()
        model = inject_lora(model, rank=4, targets=["q_proj", "v_proj"])
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        self.assertLess(trainable, total)

    def test_qlora_forward(self):
        from egx.peft.qlora import QuantizedLinear

        orig = nn.Linear(64, 32)
        q = QuantizedLinear(orig, rank=4)
        x = torch.randn(2, 64)
        out = q(x)
        self.assertEqual(out.shape, (2, 32))


if __name__ == "__main__":
    unittest.main()
