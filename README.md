# EGX — Elastic Guardian X

> **The Intelligent Adaptive Training Runtime for the Modern ML Stack.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-Alpha-orange.svg)]()

EGX (Elastic Guardian X) is an advanced, zero-configuration training runtime designed to transform experimental machine learning workflows into resilient, production-ready systems. It bridges the gap between research and deployment by providing an intelligent orchestration layer that automatically adapts to available hardware, optimizes training parameters, and ensures fault tolerance at every scale.

---

## 🚀 Quickstart & Showcases

The best way to understand EGX is to watch it work dynamically. We have provided several beautiful, interactive scripts in the `examples/` directory to demonstrate its capabilities.

### 1. The Interactive Showcase
This script provides an animated, rich terminal UI that demonstrates real-time hardware probing, strategy decision-making, and an active proxy training loop.
```bash
python examples/showcase_egx.py
```

### 2. The EGX Stress Tester
Wondering how large a model your current hardware can handle before running Out of Memory (OOM)? This script scales a model from 0.3M up to 20 Billion parameters, testing EGX's strict Layer 4 Memory Bounds safely without crashing the host OS.
```bash
python examples/stress_test_egx.py
```

### 3. Basic Example Setup
A simple standard training loop using EGX.
```bash
python examples/train_example.py
```

---

## 🏗️ Technical Architecture

EGX's architecture is built on a foundation of **7 distinct layers** and **8 specialized Data Structure & Algorithm (DSA) patterns**, following a strict modular design:

- **Intelligence Layer**: Predictive resource analysis and decision rationing.
- **Orchestration Layer**: Dynamic workflow management and task scheduling.
- **Resilience Layer**: Self-healing mechanisms and automated recovery protocols.
- **Infrastructure Layer**: Cross-platform hardware abstraction (NVIDIA/AMD/Cloud).
- **Core Engine**: The high-performance runtime that executes the training loops.

---

## 🧠 Core Capabilities

### 1. Intelligent Adaptive Runtime
EGX continuously probes your infrastructure to make real-time decisions. It analyzes VRAM availability, interconnect speeds, and compute capabilities to select the optimal training strategy (e.g., Full Fine-Tuning, LoRA, QLoRA) without manual intervention.

### 2. Zero-Configuration Orchestration
Built for speed and simplicity. A single line of code can trigger a sophisticated training pipeline that includes automated data sharding, model checkpointing, and performance monitoring.

### 3. Resilience & Fault Tolerance
With its "Inviolable Laws" engine, EGX ensures your training runs are protected against hardware failures, OOM (Out-of-Memory) errors, and network interruptions. It features a decentralized state management system that can resume training from the last healthy state with zero data loss.

### 4. Native Multi-Scale Support
Seamlessly scale from your local workstation to multi-node clusters. EGX's infrastructure layer abstracts away the complexities of distributed backend management, supporting everything from simple DP to advanced FSDP.

---

## 💻 Usage

### API Integration
EGX provides a clean, Pythonic API that fits into any existing workflow.

```python
from egx.api.trainer import EGX

# Initialize the intelligent trainer
trainer = EGX()

# Launch an adaptive training run
# EGX will auto-determine if LoRA, QLoRA, or Full FT is best for your hardware
result = trainer.train(
    model="meta-llama/Llama-2-7b-hf",
    dataset="path/to/my_data",
    max_epochs=3
)

print(f"Decision Rationale: {result.get('mode')}")
```

### CLI Interface
For engineers who prefer the terminal, EGX offers a powerful command-line interface:
```bash
# Probe current hardware capabilities
egx probe

# Launch training with a config file
egx train --config production.yaml

# Benchmark your system
egx benchmark
```

---

## 🛠️ Installation

### Quick Start
```bash
pip install egx

# Or run the Windows installer
install.bat
```

### Advanced Features
```bash
# Core installation
pip install -e .              

# Install with Flash Attention 2 support
pip install -e ".[flash]"

# Install full production suite (REST API + ONNX Export)
pip install -e ".[all]"       
```

---

## 🧪 Testing & Verification

EGX is thoroughly tested using `pytest` across several independent vectors to guarantee its resilient architecture. To run the tests, a newcomer must use the following commands:

**Run the Core Structural Diagnostic Suite:**
This runs `test_all_modules.py`, an exhaustive execution of the framework's fundamental logic layers.
```bash
python tests/test_all_modules.py
```

**Run Unit & Data Structure Integrity Tests:**
```bash
pytest tests/unit/dsa/ -v
pytest tests/unit/ -v
```

**Run the Full Framework Ecosystem Check:**
Verify absolute stability across the entire project structure.
```bash
pytest tests/
```

---

## 📄 Documentation & Resources

- **GitHub**: [hoshikrana/Elastic-Guardian](https://github.com/hoshikrana/Elastic-Guardian)
- **Technical Specification**: See `docs/architecture/EGX_Definitive_Architecture.docx`
- **Contributor Guide**: Under `docs/CONTRIBUTING.md` (Coming Soon)

## ⚖️ License
EGX is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for more details.
