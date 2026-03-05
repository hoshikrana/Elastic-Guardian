"""
EGX Test: peft/lora_plus.py
"""
import unittest
import torch.nn as nn
from egx.peft.lora import inject_lora
from egx.peft.lora_plus import get_lora_plus_param_groups

class TestLoRAPlus(unittest.TestCase):
    def test_param_groups(self):
        # Create a simple model with q_proj
        model = nn.Module()
        model.q_proj = nn.Linear(64, 64)
        model = inject_lora(model, rank=4, targets=["q_proj"])
        groups = get_lora_plus_param_groups(model, base_lr=1e-4)
        self.assertGreater(len(groups), 0)

if __name__ == "__main__":
    unittest.main()
