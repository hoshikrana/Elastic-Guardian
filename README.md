# EGX — Elastic Guardian X

**Intelligent Adaptive Training Runtime. Zero configuration. All scales. Fault-tolerant.**

EGX is designed to handle intelligent adaptive training with a focus on ease of use and scalability. It automatically handles resource allocation and optimization based on your hardware.

```python
from egx.api.trainer import EGX

result = EGX().train(model=my_model, dataset=train_dataset)
print(result.decision_rationale)  # "LoRA fits 18.2GB (75% of 24GB). Full FT needs 56GB."
```

## Features
- **GPU Adaptive**: Optimized for various GPU configurations.
- **Zero Configuration**: Ready to go out of the box.
- **Fault Tolerant**: Built-in mechanisms to handle training interruptions.
- **Scalable**: From single GPU to large-scale clusters.

## Installation
```bash
pip install egx              # core
pip install "egx[flash]"     # + FlashAttention2
pip install "egx[all]"       # + ONNX export + REST API
```

## Quick Start (Windows)
```bat
install.bat
```

## Resources
- **GitHub Repository**: [hoshikrana/Elastic-Guardian](https://github.com/hoshikrana/Elastic-Guardian)
- **Architecture**: 7 layers · 8 DSA structures · 12 inviolable laws.
- **Full Docs**: See `docs/architecture/EGX_Definitive_Architecture.docx`
