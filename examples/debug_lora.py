import torch.nn as nn
import logging
from egx.peft.lora import inject_lora

# Setup logging to see inject_lora output
logging.basicConfig(level=logging.INFO)


class DummyLlama(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": nn.ModuleDict(
                            {
                                "q_proj": nn.Linear(128, 128),
                                "v_proj": nn.Linear(128, 128),
                            }
                        )
                    }
                )
                for _ in range(2)
            ]
        )


model = DummyLlama()
print("--- Before injection ---")
for name, mod in model.named_modules():
    if isinstance(mod, nn.Linear):
        print(f"  {name}: {type(mod)}")

inject_lora(model, targets=["q_proj", "v_proj"])

print("\n--- After injection ---")
for name, mod in model.named_modules():
    if "LoRALinear" in str(type(mod)):
        print(f"  {name}: {type(mod)}")
