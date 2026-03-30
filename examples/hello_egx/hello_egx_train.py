"""
Hello EGX — Minimal Self-Contained Training Run.

This example showcases the full lifecycle of the EGX framework in a tiny,
portable script that runs on any machine (CPU or GPU).
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

from egx.api.config import EGXConfig
from egx.api.trainer import EGXTrainer
from egx.api.callbacks import LoggingCallback, CheckpointCallback

# 1. Setup Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("hello_egx")


def run_minimal_project():
    print("Starting EGX 'Hello World' Project Run...")

    # 2. Create a Tiny Architecture
    # Logic: Let's train a model to learn a simple mapping y = x * 2 + 1
    model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))

    # 3. Generate Synthetic Dataset
    # We use a simple list of dicts to be fully compatible with the EGX batch processor
    x_data = torch.randn(100, 1)
    y_data = x_data * 2 + 1 + torch.randn(100, 1) * 0.01

    dataset = [{"input": x_data[i], "labels": y_data[i]} for i in range(100)]

    # 4. EGX Configuration (Production-Ready Patterns)
    config = EGXConfig(
        learning_rate=0.01,
        num_epochs=5,
        batch_size=10,
        loss_fn="mse",  # New: leverages our hardened kernel
        output_dir="./hello_egx_output",
        save_steps=10,
        logging_steps=1,
    )

    # 5. Initialize Trainer with Hardened Callbacks
    trainer = EGXTrainer(
        config=config,
        callbacks=[
            LoggingCallback(log_every_n_steps=1),
            CheckpointCallback(save_every_n_steps=10),
        ],
    )

    # 6. Execute Training Hook (Triggers Boot -> Probing -> Training)
    print("\n--- Phase: EGX Lifecycle Execution ---")
    trainer.train(model=model, dataset=dataset)

    print("\n--- Phase: Post-Training Validation ---")
    # Test the model
    test_val = torch.tensor([[10.0]])
    with torch.no_grad():
        pred = model(test_val)
    print(f"Input: 10.0 | Expected: ~21.0 | Predicted: {pred.item():.2f}")

    # 7. Check Resilience Artifacts
    if os.path.exists("./hello_egx_output"):
        print(
            "[OK] Production artifacts (Atomic Checkpoints & Logs) created successfully."
        )
        import shutil

        shutil.rmtree("./hello_egx_output")


if __name__ == "__main__":
    run_minimal_project()
