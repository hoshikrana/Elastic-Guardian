<div align="center">

# EGX — Elastic Guardian X

**The Intelligent Adaptive Training Runtime for the Modern ML Stack.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-Production_Ready-success.svg)]()

> *Transform experimental ML workflows into resilient, self-healing, production-ready systems.*

</div>

EGX (Elastic Guardian X) is an advanced, zero-configuration training orchestrator. It acts as a resilient wrapper around PyTorch and HuggingFace, bringing dynamic hardware probing, automatic optimization routing (LoRA/Full-FT fallback), and strict memory-bounds enforcement out of the box.

If you want the flexibility of PyTorch with the reliability of a distributed enterprise system, EGX is built for you.

---

## 🚀 Quickstart & Showcases

The best way to understand EGX's intelligence is to watch it work dynamically. Clone the repo and run our pre-configured interactive showcases in the `examples/` directory!

### 1. The Production Training Demo
This script downloads `DistilGPT2`, probes your hardware (falling back to CPU/LoRA safely if you lack a GPU), and executes a beautifully logged 5-step training pipeline with self-healing features enabled.
```bash
python examples/presentation_demo.py
```

### 2. The Interactive Showcase
This script provides an animated, rich terminal UI that demonstrates real-time hardware probing, distributed strategy decision-making, and an active proxy training loop.
```bash
python examples/showcase_egx_hardening.py
```

---

## 🧠 Core Capabilities

### 1. Zero-Configuration Orchestration
You should never have to manually calculate `gradient_accumulation_steps` based on VRAM capacity again. Pass your dataset and model to `EGXTrainer` and let the engine dynamically decide if you should execute **Full Fine-Tuning** or fallback seamlessly to **LoRA/QLoRA** based on its physical runtime analysis.

### 2. Resilient "Self-Healing" Checkpointing
Using our `CheckpointManager` and `Atomic CheckpointWriter`, EGX strictly enforces **Law 1: Atomic Integrity**. Checkpoints are written to `.tmp` files alongside SHA-256 sidecars and natively renamed using `os.replace`. If your machine crashes during a 20B parameter save sequence, no corruption will ever occur. 

### 3. Transparent Callback Hierarchy
Monitor exactly what goes into the `train_step()` dynamically.
- `LoggingCallback` tracks precise step-loss and validation drops.
- `ThroughputCallback` evaluates raw token traversal bounds.
- Custom injection architectures allow you to natively pass **TypedDict Contexts** straight to your orchestrator.

---

## 💻 Usage

EGX acts as a transparent wrapper. It natively accepts standard HuggingFace Models and Tokenizers!

### Basic Configuration
```python
from transformers import AutoModelForCausalLM
from egx.api.config import EGXConfig
from egx.api.trainer import EGXTrainer

# 1. Load standard models
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# 2. Tell EGX what your boundaries are
config = EGXConfig(
    batch_size=8,
    max_steps=500,
    learning_rate=5e-5,
    # EGX will detect memory caps and automatically inject 
    # LoRA bindings into `model` if needed!
)

# 3. Train flawlessly with callbacks
trainer = EGXTrainer(config=config)
result = trainer.train(model, dataset)

print(f"Time Taken: {result['duration_s']}s")
```

### Custom Training Loops (Method 2)
Need total control? Write a custom step function, but keep the hardware optimizations:
```python
def contrastive_loss_step(model, batch, device) -> float:
    # Do whatever complex math you want here!
    inputs = batch["input_ids"].to(device)
    loss = complex_custom_forward(model, inputs)
    loss.backward()
    return loss.item()

# Inject it into the trainer! EGX still handles epoch logging,
# saving, loading, device mounting, and mixed-precision tracking.
trainer = EGXTrainer(config=config, training_step_fn=contrastive_loss_step)
trainer.train(model, dataset)
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/hoshikrana/Elastic-Guardian.git
cd Elastic-Guardian

# Install python dependencies efficiently!
pip install -e .

# OR use our direct bootstrapper
./install.bat
```

---

## 🧪 Testing & Verification

EGX is thoroughly tested using an exhaustive, 360-assertion `pytest` matrix to guarantee 100% architectural adherence to the Core Design Laws. It evaluates Memory Assertions, Pydantic type safety, and FSM (Finite State-Machine) crash loops.

To verify your installation across all vectors:
```bash
python scripts/run_tests.py
```
*Or natively via pytest:*
```bash
pytest tests/ -x --tb=short
```

---

## 📄 Documentation & Architecture
- **Technical Specification**: Dive deeper into the runtime state machines in `docs/audit_reports/CODEQUALITYAUDIT_PHASE2.md`.
- **System Architecture**: Read about the 10 Immutable Laws in `docs/audit_reports/COMPREHENSIVE_SENIOR_REVIEW.md`.

## ⚖️ License
EGX is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for more details.
