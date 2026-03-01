# ADR-004: export/ as Layer 5 Module
Status: ACCEPTED

## Problem
Old elastic_guardian had export/ package. EGX v1 omitted it.
Without export, training has no deployment path.
LoRA merge is critical: deployed model MUST have adapters merged into base weights.

## Decision
Add egx/export/ at Layer 5. Responsibilities:
- lora_merger.py: merge LoRA/QLoRA/DoRA into base (calls peft/merger.py)
- safetensors_exporter.py: default export format, sharded for large models
- onnx_exporter.py: optional ONNX export (dep: onnx + onnxruntime)
- base_exporter.py: ABC for all exporters

## Impact
TrainingResult gains optional export_result field.
CLI gains `egx export` subcommand.
safetensors added to core deps. onnx added as optional dep.
