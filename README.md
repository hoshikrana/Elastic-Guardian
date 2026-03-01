# EGX — Elastic Guardian X

**Intelligent Adaptive Training Runtime. Zero configuration. All scales. Fault-tolerant.**

```python
from egx.api.trainer import EGX

result = EGX().train(model=my_model, dataset=train_dataset)
print(result.decision_rationale)  # "LoRA fits 18.2GB (75% of 24GB). Full FT needs 56GB."
```

## Installation
```bash
pip install egx              # core
pip install "egx[flash]"     # + FlashAttention2
pip install "egx[all]"       # + ONNX export + REST API
```

## Windows
```bat
install.bat
```

Architecture: 7 layers · 8 DSA structures · 12 inviolable laws.
See `docs/architecture/EGX_Definitive_Architecture.docx`
