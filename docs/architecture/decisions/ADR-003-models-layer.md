# ADR-003: models/ as Layer 5 Module
Status: ACCEPTED

## Problem
Old elastic_guardian had models/ package. EGX v1 omitted it.
ModelProfile (core/models.py) is a dataclass but nothing loads or introspects a real nn.Module.
Phase 4 of lifecycle.py ("Model Introspection") had no concrete implementation.

## Decision
Add egx/models/ at Layer 5. Responsibilities:
- auto_detect.py: architecture from config.json / layer name patterns
- introspector.py: live nn.Module -> ModelProfile (param count, hidden_dim, layers)
- loader.py: HF Hub / local path -> (nn.Module, ModelProfile)
- registry.py: known arch metadata + default LoRA targets

## Impact
lifecycle.py Phase 4 delegates to models/introspector.py.
peft/injector.py target detection delegates to models/introspector.py.
