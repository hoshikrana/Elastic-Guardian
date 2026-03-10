"""GPU Validation: LoRA merge accuracy tests."""
import unittest
import torch
import torch.nn as nn


class TestLoRAMergeAccuracy(unittest.TestCase):
    def test_lora_output_shape(self):
        from egx.peft.lora import LoRALinear
        orig = nn.Linear(64, 32)
        lora = LoRALinear(orig, rank=4, alpha=8)
        x = torch.randn(4, 64)
        out = lora(x)
        self.assertEqual(out.shape, (4, 32))

    def test_lora_trainable_params_match(self):
        from egx.peft.lora import LoRALinear
        orig = nn.Linear(64, 32)
        lora = LoRALinear(orig, rank=8)
        expected = 8 * 64 + 32 * 8
        self.assertEqual(lora.trainable_params, expected)

    def test_inject_lora_preserves_non_targets(self):
        from egx.peft.lora import inject_lora
        from tests.mocks.mock_model import ModelWithQProj
        model = ModelWithQProj()
        model = inject_lora(model, rank=4, targets=["q_proj"])
        self.assertIsInstance(model.out_proj, nn.Linear)


if __name__ == "__main__":
    unittest.main()
