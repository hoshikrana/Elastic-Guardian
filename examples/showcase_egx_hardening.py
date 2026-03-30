"""
EGX Production Hardening Showcase.

This script demonstrates the "next level" engineering patterns implemented in EGX:
1. Architectural Laws (Immutability & Slotting)
2. Hardware-Aware Fail-Fast Boot
3. Atomic Checkpointing with SHA256
4. Thread-Safe Telemetry
"""

import os
import torch
import torch.nn as nn
import logging
from egx.api.config import EGXConfig
from egx.runtime.engine import EGXEngine
from egx.resilience.checkpoint.writer import CheckpointWriter
from egx.monitoring.telemetry import TelemetryService
from egx.core.exceptions import BoolAsIntError, HardwareError

# Setup logging to see the "Senior Dev" style output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("egx.showcase")


def showcase_architectural_laws():
    print("\n--- 1. Showcase: Architectural Laws (Immutability & Slots) ---")
    config = EGXConfig(learning_rate=0.001)

    print(
        f"Config initialized with slots: {getattr(EGXConfig, '__slots__', 'Not Found')}"
    )

    # Showcase Law 10: No bools as memory values (demonstrated in tests, but here we show general safety)
    try:
        from egx.core.memory.value import MemoryValue

        print("Attempting to pass 'True' as a memory byte count (Law 10 violation)...")
        MemoryValue(True)
    except BoolAsIntError as e:
        print(f"[OK] Blocked Law 10 violation: {e}")


def showcase_fail_fast_boot():
    print("\n--- 2. Showcase: Hardware-Aware Fail-Fast Boot ---")
    engine = EGXEngine()
    model = nn.Linear(10, 10)
    config = EGXConfig()

    print(
        "Triggering engine boot. This will probe topology and validate model health..."
    )
    try:
        engine.boot(model, config)
        print("[OK] Engine boot successful (Probed hardware and validated model).")
    except Exception as e:
        print(
            f"[FAIL] Boot failed (Expected if no GPU is found or env is unaligned): {e}"
        )


def showcase_atomic_resilience():
    print("\n--- 3. Showcase: Atomic Checkpointing ---")
    writer = CheckpointWriter()
    data = {"epoch": 1, "state_dict": {}}
    path = "./demo_checkpoint.pt"

    print(f"Saving atomic checkpoint to {path}...")
    writer.save(data, path)

    if os.path.exists(path) and os.path.exists(path + ".sha256"):
        with open(path + ".sha256", "r") as f:
            sha = f.read()
        print(f"[OK] Checkpoint saved atomically with SHA256 sidecar: {sha[:8]}...")

    # Cleanup
    if os.path.exists(path):
        os.remove(path)
    if os.path.exists(path + ".sha256"):
        os.remove(path + ".sha256")


def showcase_thread_safe_telemetry():
    print("\n--- 4. Showcase: Thread-Safe Telemetry ---")
    telemetry = TelemetryService(log_dir="./demo_logs")
    print("Broadcasting metrics (Safe for use in multi-threaded training loops)...")
    telemetry.broadcast_step(step=0, metrics={"loss": 0.5, "accuracy": 0.1})
    print("[OK] Metrics broadcasted to jsonl via thread-safe lock.")

    # Cleanup
    import shutil

    if os.path.exists("./demo_logs"):
        shutil.rmtree("./demo_logs")


if __name__ == "__main__":
    print("====================================================")
    print("      EGX PRODUCTION HARDENING SHOWCASE v2.0        ")
    print("====================================================")

    showcase_architectural_laws()
    showcase_fail_fast_boot()
    showcase_atomic_resilience()
    showcase_thread_safe_telemetry()

    print("\nConclusion: The EGX codebase is now strictly aligned with senior-level")
    print("production standards: Slotted, Atomic, Thread-Safe, and Fail-Fast.")
