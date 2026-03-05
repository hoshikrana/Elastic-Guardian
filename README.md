# EGX — Elastic Guardian X

> **The Intelligent Adaptive Training Runtime for the Modern ML Stack.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-Alpha-orange.svg)]()

EGX (Elastic Guardian X) is an advanced, zero-configuration training runtime designed to transform experimental machine learning workflows into resilient, production-ready systems. It bridges the gap between research and deployment by providing an intelligent orchestration layer that automatically adapts to available hardware, optimizes training parameters, and ensures fault tolerance at every scale.

---

## 🚀 Vision

In modern ML engineering, the transition from a local notebook to a large-scale GPU cluster is often fraught with boilerplate code, complex configuration, and fragile pipelines. **EGX eliminates this friction.**

It treats hardware as a dynamic resource, not a static constraint. Whether you are fine-tuning a 7B parameter model on a single consumer GPU or running massive distributed training, EGX intelligently manages the underlying complexity, allowing you to focus on the science of machine learning.

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

## 🏗️ Technical Architecture

EGX's architecture is built on a foundation of **7 distinct layers** and **8 specialized Data Structure & Algorithm (DSA) patterns**, following a strict modular design:

- **Intelligence Layer**: Predictive resource analysis and decision rationing.
- **Orchestration Layer**: Dynamic workflow management and task scheduling.
- **Resilience Layer**: Self-healing mechanisms and automated recovery protocols.
- **Infrastructure Layer**: Cross-platform hardware abstraction (NVIDIA/AMD/Cloud).
- **Core Engine**: The high-performance runtime that executes the training loops.

---

## 💻 Usage

### API Integration
EGX provides a clean, Pythonic API that fits into any existing workflow.
**Intelligent Adaptive Training Runtime. Zero configuration. All scales. Fault-tolerant.**

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

print(f"Decision Rationale: {result.decision_rationale}")
result = EGX().train(model=my_model, dataset=train_dataset)
print(result["mode"])  # "LoRA fits 18.2GB (75% of 24GB). Full FT needs 56GB."
```

### CLI Interface
For engineers who prefer the terminal, EGX offers a powerful command-line interface:

## Installation
```bash
# Probe current hardware capabilities
egx probe

# Launch training with a config file
egx train --config production.yaml

# Benchmark your system
egx benchmark
pip install -e .              # core
pip install -e ".[flash]"     # + FlashAttention2
pip install -e ".[all]"       # + ONNX export + REST API
```

---

## 🛠️ Installation

### Quick Start
```bash
pip install egx
## Windows
```bat
install.bat
```

### Advanced Features
```bash
# Install with Flash Attention 2 support
pip install "egx[flash]"

# Install full production suite (REST API + ONNX Export)
pip install "egx[all]"
```

---

## 📄 Documentation & Resources

- **GitHub**: [hoshikrana/Elastic-Guardian](https://github.com/hoshikrana/Elastic-Guardian)
- **Technical Specification**: See `docs/architecture/EGX_Definitive_Architecture.docx`
- **Contributor Guide**: Under `docs/CONTRIBUTING.md` (Coming Soon)

---

## ⚖️ License

EGX is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for more details.
Architecture: 7 layers · 8 DSA structures · 12 inviolable laws.
See `docs/architecture/`
