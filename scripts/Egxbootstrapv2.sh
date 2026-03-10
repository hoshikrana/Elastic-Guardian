#!/usr/bin/env bash
# =============================================================================
# EGX — Elastic Guardian X
# Complete Directory & File Structure Bootstrap  v2.0
#
# CHANGES FROM v1 (gap analysis against old elastic_guardian repo):
#   + egx/models/       — model registry, loader, introspector, arch-detect
#   + egx/export/       — ONNX, safetensors, LoRA merger (missing deploy path)
#   + egx/peft/merger.py — LoRA weight merge-back (critical, was missing)
#   + Fixed egx/__init__.py — removed Layer 7 import (was Law 4 violation)
#   + install.bat / install.sh — developer installers (you're on Windows)
#   + 12 new test files for models/ + export/ pipelines
#   + 4 ADRs documenting all new architecture decisions
#
# Usage:
#   Linux/macOS:  bash egx_bootstrap_v2.sh
#   Windows:      Run install.bat after this generates it,
#                 or use Git Bash / WSL to run this script directly
# =============================================================================

set -euo pipefail
ROOT="egx"

echo ""
echo "  EGX — Elastic Guardian X"
echo "  Bootstrap v2.0 — Creating complete project structure..."
echo ""

mkf() {
  local path="$1"; local comment="$2"
  mkdir -p "$(dirname "$path")"
  printf '# %s\n' "$comment" > "$path"
}

# =============================================================================
# ROOT PROJECT FILES
# =============================================================================
mkdir -p "$ROOT"

cat > "pyproject.toml" << 'EOF'
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "egx"
version = "0.1.0"
description = "Elastic Guardian X — Intelligent Adaptive Training Runtime"
readme = "README.md"
license = { text = "Apache-2.0" }
requires-python = ">=3.10"
keywords = ["machine-learning", "training", "gpu", "adaptive", "lora", "qlora"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "torch>=2.1.0",
    "pynvml>=11.5.0",
    "pydantic>=2.0.0",
    "click>=8.1.0",
    "pyyaml>=6.0",
    "structlog>=23.0.0",
    "bitsandbytes>=0.41.0",
    "safetensors>=0.4.0",
]

[project.optional-dependencies]
dev   = [
    "pytest>=7.4", "pytest-cov>=4.1", "pytest-benchmark>=4.0",
    "pytest-timeout>=2.2", "mypy>=1.7", "ruff>=0.1",
    "black>=23.0", "hypothesis>=6.0",
]
flash = ["flash-attn>=2.3.0"]
onnx  = ["onnx>=1.15.0", "onnxruntime>=1.16.0"]
api   = ["fastapi>=0.100", "uvicorn>=0.24"]
all   = ["egx[flash,onnx,api]"]

[project.urls]
Repository = "https://github.com/hoshikrana/Elastic-Guardian"

[project.scripts]
egx = "egx.cli.main:main"

[tool.mypy]
strict = true
disallow_any_explicit = true
warn_return_any = true
warn_unused_ignores = true

[tool.ruff]
line-length = 100
target-version = "py310"
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]

[tool.black]
line-length = 100
target-version = ["py310", "py311", "py312"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short --timeout=120"
markers = [
    "gpu: requires CUDA GPU",
    "slow: takes more than 30s",
    "benchmark: performance benchmarks",
]
EOF

cat > "README.md" << 'EOF'
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
EOF

cat > ".gitignore" << 'EOF'
__pycache__/
*.py[cod]
*.pyd
*.so
*.dll
.env
.venv
venv/
dist/
build/
*.egg-info/
.mypy_cache/
.ruff_cache/
.pytest_cache/
htmlcov/
.coverage
*.db
*.log
egx_runs/
checkpoints/
exports/
*.onnx
*.safetensors
.idea/
.vscode/
.DS_Store
Thumbs.db
EOF

echo "3.11" > ".python-version"

cat > "install.bat" << 'EOF'
@echo off
REM EGX — Elastic Guardian X — Windows Installer
echo EGX Elastic Guardian X — Windows Install
echo.
python --version >nul 2>&1
IF ERRORLEVEL 1 (echo ERROR: Python not found. Install Python 3.10+ first. & pause & exit /b 1)
echo [1/4] Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip
echo [3/4] Installing EGX (dev mode)...
pip install -e ".[dev]"
echo [4/4] Verifying...
python -c "import egx; print('EGX', egx.__version__, 'OK')"
echo.
echo Done. Activate: .venv\Scripts\activate.bat
pause
EOF

cat > "install.sh" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "EGX — Install"
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e ".[dev]"
python -c "import egx; print('EGX', egx.__version__, 'OK')"
EOF
chmod +x install.sh

# PRE-CREATE ALL DIRECTORIES
mkdir -p "$ROOT/core/memory" "$ROOT/infrastructure" "$ROOT/intelligence/estimator/calibration"
mkdir -p "$ROOT/intelligence/planner" "$ROOT/intelligence/strategy" "$ROOT/intelligence/graph"
mkdir -p "$ROOT/orchestration/executor" "$ROOT/orchestration/pressure" "$ROOT/orchestration/swapper"
mkdir -p "$ROOT/resilience/checkpoint" "$ROOT/models" "$ROOT/export"
mkdir -p "$ROOT/training" "$ROOT/peft" "$ROOT/data" "$ROOT/monitoring"
mkdir -p "$ROOT/runtime" "$ROOT/api" "$ROOT/cli" "$ROOT/plugins"

# =============================================================================
# LAYER 1: core/  — stdlib only, zero external deps
# =============================================================================
cat > "$ROOT/core/__init__.py" << 'EOF'
# core — Layer 1. Re-exports only. Zero external dependencies. No torch.
EOF
mkf "$ROOT/core/enums.py"              "All EGX enumerations. stdlib only. str-typed for JSON serialization."
mkf "$ROOT/core/exceptions.py"         "Typed exception hierarchy (20+ types). Every exception: recoverable: bool explicit."
mkf "$ROOT/core/models.py"             "All frozen dataclasses. frozen=True, slots=True. No torch. No third-party deps."
mkf "$ROOT/core/constants.py"          "Constants only: KB/MB/GB/TB. No logic, no classes, no functions."
mkf "$ROOT/core/memory/__init__.py"    "core.memory — Memory value object and pure utility functions."
mkf "$ROOT/core/memory/value.py"       "Memory immutable value object. int bytes only. sys.maxsize overflow guard."
mkf "$ROOT/core/memory/validators.py"  "MemoryValidator ABC. bool check MUST precede int check (isinstance(True,int)==True trap)."
mkf "$ROOT/core/memory/units.py"       "convert_to_bytes(), convert_from_bytes(). Pure functions. All return int. Validator injected."
mkf "$ROOT/core/memory/formatting.py"  "format_bytes() pure display function. No state. No validation. No side effects."

# =============================================================================
# LAYER 2: infrastructure/  — torch + pynvml allowed here only
# =============================================================================
cat > "$ROOT/infrastructure/__init__.py" << 'EOF'
# infrastructure — Layer 2. Hardware access. torch + pynvml allowed only here and L5+.
EOF
mkf "$ROOT/infrastructure/gpu_probe.py"        "NVML -> nvidia-smi (subprocess, timeout=5s) -> /proc fallback. Returns list[GPUSpec]."
mkf "$ROOT/infrastructure/topology_builder.py" "Assembles HardwareTopology. NVLink via nvmlDeviceGetP2PStatus. Measured bandwidth."
mkf "$ROOT/infrastructure/torch_runner.py"     "Isolated dry-run. 2 warmup + 3 measured steps. ThreadPoolExecutor timeout=45s."
mkf "$ROOT/infrastructure/nvme_probe.py"       "NVMe sequential R/W benchmark. 128MB temp file. 3 runs. Returns int bytes/s."
mkf "$ROOT/infrastructure/bandwidth_sampler.py" "PCIe/NVLink bandwidth. Timed tensor transfers. 3 warmup + 5 measured. Returns int bytes/s."
mkf "$ROOT/infrastructure/structured_logger.py" "JSON structured logging. PID-scoped singleton. Handler dedup guard. Never print()."

# =============================================================================
# LAYER 3: intelligence/  — pure logic, no torch, 8 DSA structures here
# =============================================================================
cat > "$ROOT/intelligence/__init__.py" << 'EOF'
# intelligence — Layer 3. Pure decision logic. No torch. No pynvml.
# Houses all 8 DSA structures.
EOF
cat > "$ROOT/intelligence/estimator/__init__.py" << 'EOF'
# intelligence.estimator — 3-method estimation pipeline (analytical/dryrun/hybrid).
EOF
mkf "$ROOT/intelligence/estimator/base.py"       "BaseEstimator ABC. estimate(topo, model, plan) -> MemoryReport. No concrete logic."
mkf "$ROOT/intelligence/estimator/analytical.py" "Formula estimator. 5 modes x 4 optimizer types. <1ms. All results int bytes."
mkf "$ROOT/intelligence/estimator/dryrun.py"     "DSA-4: MemorySegmentTree (range-max, O(log n)). Measurement-based. Timeout=45s."
mkf "$ROOT/intelligence/estimator/hybrid.py"     "Blend dryrun(0.70)+analytical(0.30). CalibrationCache O(1) hot path. Log fallbacks."
cat > "$ROOT/intelligence/estimator/calibration/__init__.py" << 'EOF'
# calibration — DSA-2: Red-Black Tree (store) + DSA-2b: LRU Cache (cache).
EOF
mkf "$ROOT/intelligence/estimator/calibration/cache.py"      "DSA-2b: LRU via OrderedDict. O(1) get/put. move_to_end on hit. Capacity-bounded."
mkf "$ROOT/intelligence/estimator/calibration/store.py"      "DSA-2a: Red-Black Tree + SQLite. O(log n) range-query by similar hardware fingerprint."
mkf "$ROOT/intelligence/estimator/calibration/regression.py" "Phase 2: polynomial regression on calibration history. ML-corrected estimates."

cat > "$ROOT/intelligence/planner/__init__.py" << 'EOF'
# intelligence.planner — Memory planning and tensor allocation.
EOF
mkf "$ROOT/intelligence/planner/memory_planner.py"    "Per-strategy VRAM footprint. Calls analytical estimator. Feeds selector."
mkf "$ROOT/intelligence/planner/allocation_planner.py" "Tensor placement + timing-aware prefetch schedule. Dijkstra for tensor routing."
mkf "$ROOT/intelligence/planner/timeline_planner.py"  "transfer_ms=size/(bw*1e9)*1000. lead_steps=ceil(ms/step_ms). Computed, never guessed."
mkf "$ROOT/intelligence/planner/topology_planner.py"  "Multi-GPU parallel strategy from topology. Delegates to parallel_advisor."

cat > "$ROOT/intelligence/strategy/__init__.py" << 'EOF'
# intelligence.strategy — DSA-1: Fibonacci Heap (selector). DSA-7: Binary Search (batch).
EOF
mkf "$ROOT/intelligence/strategy/selector.py"         "DSA-1: Fibonacci Heap. O(1) amortized insert/decrease-key. O(log n) extract-max."
mkf "$ROOT/intelligence/strategy/scorer.py"           "Composite: memory_safety=0.40, speed=0.25, param_eff=0.20, user_pref=0.15. Sum=1.0."
mkf "$ROOT/intelligence/strategy/batch_optimizer.py"  "DSA-7: Binary search. O(log n). Monotone predicate: fits(N) implies fits(N-1)."
mkf "$ROOT/intelligence/strategy/parallel_advisor.py" "FSDP/TP/PP/3D from topology effective bandwidth. Dijkstra gates the decision."

cat > "$ROOT/intelligence/graph/__init__.py" << 'EOF'
# intelligence.graph — DSA-5: Dijkstra. DSA-6: Kahn's BFS.
EOF
mkf "$ROOT/intelligence/graph/dependency_dag.py" "DSA-6: Kahn's BFS. Module import cycle detection at startup. O(V+E). Fails fast."
mkf "$ROOT/intelligence/graph/topology_graph.py" "DSA-5: Dijkstra on weighted HW graph. Min-latency tensor routing. O((V+E)logV)."

# =============================================================================
# LAYER 4a: orchestration/  — DSA-3: Skip List in pressure/monitor.py
# =============================================================================
cat > "$ROOT/orchestration/__init__.py" << 'EOF'
# orchestration — Layer 4a. Memory execution and pressure management.
# DSA-3: Skip List in pressure/monitor.py (lock-free concurrent event log).
EOF
cat > "$ROOT/orchestration/executor/__init__.py" << 'EOF'
# orchestration.executor
EOF
mkf "$ROOT/orchestration/executor/allocation_exec.py" "Execute AllocationPlan tensor-by-tensor. VRAM drift check: >10% triggers re-plan."
mkf "$ROOT/orchestration/executor/prefetch_exec.py"   "Execute PrefetchSchedule on CUDA streams. Lead-time enforced. Stream timeout."
mkf "$ROOT/orchestration/executor/stream_manager.py"  "CUDA stream lifecycle. One stream per schedule ID. Force-close on timeout. No leaks."
cat > "$ROOT/orchestration/pressure/__init__.py" << 'EOF'
# orchestration.pressure
EOF
mkf "$ROOT/orchestration/pressure/monitor.py"        "DSA-3: PressureEventSkipList. O(log n) concurrent insert/search. 5-threshold model."
mkf "$ROOT/orchestration/pressure/elastic_batch.py"  "OOM: batch//2 immediately. Stable 3 steps: x1.05. Stable 6 steps: x1.10."
mkf "$ROOT/orchestration/pressure/eviction_policy.py" "LRU + priority eviction. Which tensors swap to RAM/NVMe under pressure."
cat > "$ROOT/orchestration/swapper/__init__.py" << 'EOF'
# orchestration.swapper
EOF
mkf "$ROOT/orchestration/swapper/vram_to_ram.py" "VRAM->RAM via pinned memory. Non-blocking copy. Transfer time tracked vs TensorPlacement."
mkf "$ROOT/orchestration/swapper/ram_to_nvme.py" "RAM->NVMe via mmap. Zero-copy reload. Async write with fsync before VRAM release."

# =============================================================================
# LAYER 4b: resilience/
# =============================================================================
cat > "$ROOT/resilience/__init__.py" << 'EOF'
# resilience — Layer 4b. 9 failure types. Typed FSM. All transitions logged.
EOF
mkf "$ROOT/resilience/watchdog.py"      "Daemon thread. heartbeat() each step. 30s silence -> DeadlockError -> RecoveryFSM."
mkf "$ROOT/resilience/recovery_fsm.py"  "Typed FSM. 9 failures. States: HEALTHY/DEGRADED/RECOVERING/ESCALATED/ABORTED."
mkf "$ROOT/resilience/sanitizer.py"     "Per-tensor NaN/Inf after each backward. InfGradientError carries layer_name."
cat > "$ROOT/resilience/checkpoint/__init__.py" << 'EOF'
# resilience.checkpoint — Atomic write + SHA256 integrity + fallback chain.
EOF
mkf "$ROOT/resilience/checkpoint/manager.py" "Adaptive: LOSS_BASED/TIME_BASED/STEP_BASED/ADAPTIVE. Retention policy enforced."
mkf "$ROOT/resilience/checkpoint/writer.py"  "Atomic: .tmp -> fsync -> rename. SHA256 alongside. Never overwrite in-place."
mkf "$ROOT/resilience/checkpoint/reader.py"  "SHA256 verify before load. Fallback: latest -> previous -> oldest valid."

# =============================================================================
# LAYER 5a: models/  [NEW — from old elastic_guardian/models/]
# Auto-detect architecture, load from HF/local, produce ModelProfile.
# =============================================================================
cat > "$ROOT/models/__init__.py" << 'EOF'
# models — Layer 5a. Model registry, loader, and architecture introspection.
# NEW: derived from old elastic_guardian/models/ analysis.
# Produces ModelProfile consumed by intelligence layer (L3).
EOF
mkf "$ROOT/models/registry.py"     "Known arch registry: llama/mistral/falcon/gpt2/t5/bert -> arch metadata + default targets."
mkf "$ROOT/models/loader.py"       "Load: HF Hub path / local .safetensors / local .bin -> (nn.Module, ModelProfile)."
mkf "$ROOT/models/introspector.py" "Live nn.Module -> ModelProfile. Param count, hidden_dim, layers, heads via name heuristics."
mkf "$ROOT/models/auto_detect.py"  "Arch auto-detect: config.json / model card / layer name patterns -> ArchType enum."

# =============================================================================
# LAYER 5b: export/  [NEW — from old elastic_guardian/export/]
# LoRA merge + safetensors + ONNX. The missing deployment path.
# =============================================================================
cat > "$ROOT/export/__init__.py" << 'EOF'
# export — Layer 5b. Post-training export for deployment.
# NEW: derived from old elastic_guardian/export/ analysis.
# LoRA merge is CRITICAL — unmerged adapters cannot be deployed.
EOF
mkf "$ROOT/export/base_exporter.py"        "BaseExporter ABC. export(model, plan, output_dir) -> ExportResult dataclass."
mkf "$ROOT/export/lora_merger.py"          "CRITICAL: merge LoRA/QLoRA/DoRA weights into base before deployment. Calls peft/merger.py."
mkf "$ROOT/export/safetensors_exporter.py" "Export to safetensors. Sharded export for models >10GB. Default format."
mkf "$ROOT/export/onnx_exporter.py"        "ONNX export with dynamic axes. Optional dep (onnx+onnxruntime). Graceful skip if absent."

# =============================================================================
# LAYER 5c: training/
# =============================================================================
cat > "$ROOT/training/__init__.py" << 'EOF'
# training — Layer 5c. Self-healing training loop.
EOF
mkf "$ROOT/training/kernel.py"                "Main loop. heartbeat() every step. All EGXErrors -> RecoveryFSM. Never bare except."
mkf "$ROOT/training/elastic_loop.py"          "Outer retry loop. Budget per failure type. TrainingResult on any exit."
mkf "$ROOT/training/mixed_precision.py"       "AMP + GradScaler. Auto-disable on overflow. Restore on NaN recovery."
mkf "$ROOT/training/gradient_accumulation.py" "Accumulation with elastic batch sync. Correct effective-batch on mid-epoch resize."
mkf "$ROOT/training/throughput_tracker.py"    "tokens/sec rolling window N=50. Step time histogram. ETA to max_steps."

# =============================================================================
# LAYER 5d: peft/
# =============================================================================
cat > "$ROOT/peft/__init__.py" << 'EOF'
# peft — Layer 5d. PEFT injection and merge.
EOF
mkf "$ROOT/peft/injector.py"  "Reads TrainingPlan.lora_targets. Injects adapters. Logs every detected target."
mkf "$ROOT/peft/lora.py"      "LoRA rank decomposition A*B. Trainable A and B. Alpha scaling. Merge on export."
mkf "$ROOT/peft/qlora.py"     "QLoRA: INT4 base via bitsandbytes + BF16 adapters. Optimizer: P_lora only (NOT P)."
mkf "$ROOT/peft/lora_plus.py" "LoRA+: asymmetric LR. B: lr*lambda (default 16). A: lr. Better convergence."
mkf "$ROOT/peft/dora.py"      "DoRA: magnitude + direction decomp. Better convergence than standard LoRA."
mkf "$ROOT/peft/merger.py"    "Merge adapter weights into base. Called by export/lora_merger.py. Required pre-deploy."

# =============================================================================
# LAYER 5e: data/
# =============================================================================
cat > "$ROOT/data/__init__.py" << 'EOF'
# data — Layer 5e. NVMe-aware data pipeline.
EOF
mkf "$ROOT/data/loader.py"     "NVMe-aware DataLoader. Sets num_workers/pin_memory/prefetch_factor from topology."
mkf "$ROOT/data/prefetcher.py" "Async prefetch. asyncio.Queue maxsize=prefetch_factor. Never blocks training thread."
mkf "$ROOT/data/streaming.py"  "Large dataset streaming. Memory-mapped. Chunk-based for TB-scale datasets."
mkf "$ROOT/data/collator.py"   "Dynamic padding collator. Sequence bucketing reduces padding waste 30-50%."

# =============================================================================
# LAYER 5f: monitoring/
# =============================================================================
cat > "$ROOT/monitoring/__init__.py" << 'EOF'
# monitoring — Layer 5f. Metrics, anomaly detection, opt-in telemetry.
# TELEMETRY DEFAULT = FALSE. Never change this default.
EOF
mkf "$ROOT/monitoring/metrics.py"           "Step metrics: loss/throughput/VRAM/grad_norm. JSON via structured_logger."
mkf "$ROOT/monitoring/anomaly_detection.py" "Loss spike (>3sigma). Grad norm outlier (>4sigma). Alert via logger only."
mkf "$ROOT/monitoring/telemetry.py"         "OPT-IN ONLY. enabled=False always. Explicit user consent required."

# =============================================================================
# LAYER 6: runtime/  — DSA-8: Trie + HashMap
# =============================================================================
cat > "$ROOT/runtime/__init__.py" << 'EOF'
# runtime — Layer 6. System coordination. DSA-8: Trie (config) + HashMap (plugins).
EOF
mkf "$ROOT/runtime/engine.py"          "EGXEngine: DI container. Wires all layers. Runs 10-phase lifecycle."
mkf "$ROOT/runtime/lifecycle.py"       "10 phases: DAG->probe->topo->model->estimate->strategy->allocate->peft->exec->train."
mkf "$ROOT/runtime/plugin_registry.py" "DSA-8b: HashMap O(1). Thread-safe RLock. register/get/list/conflict-detect."
mkf "$ROOT/runtime/config_loader.py"   "DSA-8a: Trie. O(k) lookup + O(k+n) prefix queries. CLI autocomplete backed by this."

# =============================================================================
# LAYER 7: api/ + cli/
# =============================================================================
cat > "$ROOT/api/__init__.py" << 'EOF'
# api — Layer 7. Public surface. Frozen from v0.1.
EOF
mkf "$ROOT/api/trainer.py" "EGX class. .train(model, dataset, **kwargs) -> TrainingResult. Zero required args."
mkf "$ROOT/api/config.py"  "EGXConfig dataclass. All optional. Never required. Override only what you need."

cat > "$ROOT/cli/__init__.py" << 'EOF'
# cli — Layer 7.
EOF
mkf "$ROOT/cli/main.py" "CLI: egx train / egx probe / egx benchmark / egx config / egx export."

# =============================================================================
# PLUGINS/
# =============================================================================
cat > "$ROOT/plugins/__init__.py" << 'EOF'
# plugins — Optional extensions. Auto-registered if dep is available.
EOF
mkf "$ROOT/plugins/flash_attention.py"        "FlashAttention2. Auto-registers when flash-attn detected."
mkf "$ROOT/plugins/zero3.py"                  "DeepSpeed ZeRO-3. For >8 GPU setups."
mkf "$ROOT/plugins/cpu_offload.py"            "CPU optimizer offload. Adam moments to pinned CPU RAM."
mkf "$ROOT/plugins/gradient_checkpointing.py" "Gradient checkpointing. torch.utils.checkpoint with selective config."

# =============================================================================
# FIXED: egx/__init__.py  (v1 had Law 4 violation — importing Layer 7 from here)
# =============================================================================
cat > "$ROOT/__init__.py" << 'EOF'
"""
EGX — Elastic Guardian X  v0.1.0
Intelligent Adaptive Training Runtime.

Quickstart:
    from egx.api.trainer import EGX
    from egx.api.config import EGXConfig
    result = EGX().train(model=my_model, dataset=train_dataset)

Architecture: 7 layers, 8 DSA structures, 12 inviolable laws.
See: docs/architecture/EGX_Definitive_Architecture.docx
"""

__version__ = "0.1.0"
__author__ = "Hoshik Rana"
__license__ = "Apache-2.0"

# NO imports from egx.api or any other subpackage here.
# Importing Layer 7 (api/) from the package root violates Law 4.
# Import explicitly: from egx.api.trainer import EGX
EOF

# =============================================================================
# CONFIG/
# =============================================================================
mkdir -p "config"
cat > "config/default.yaml" << 'EOF'
# EGX default configuration — Trie-resolved by runtime/config_loader.py

intelligence:
  estimator:
    analytical:
      safety_thresholds:
        full_finetune: 0.72
        lora_plus:     0.80
        lora:          0.80
        dora:          0.82
        qlora:         0.90
    hybrid:
      dryrun_weight:        0.70
      analytical_weight:    0.30
      dryrun_timeout_s:     45.0
      confidence_threshold: 0.85
    calibration:
      max_cache_size: 1024
      retention_days: 90
  strategy:
    scoring_weights:
      memory_safety:    0.40
      training_speed:   0.25
      param_efficiency: 0.20
      user_preference:  0.15
    batch_search:
      low: 1
      high: 512

orchestration:
  pressure:
    poll_interval_s: 1.0
    thresholds:
      green:     0.72
      yellow:    0.85
      orange:    0.92
      red:       0.97
      emergency: 0.99
  elastic_batch:
    stability_window: 3
    growth_phase1:    1.05
    growth_phase2:    1.10

resilience:
  watchdog:
    heartbeat_timeout_s: 30.0
  recovery:
    max_oom_retries:      5
    max_nan_retries:      10
    max_inf_retries:      5
    max_deadlock_retries: 3
  checkpoint:
    strategy:          adaptive
    time_interval_min: 30
    step_interval:     500
    keep_top_n_loss:   2
    keep_last_n_time:  2

models:
  auto_detect:
    fallback_to_heuristics: true
    log_detected_targets:   true

export:
  default_format:          safetensors
  lora_merge_before_export: true
  onnx:
    opset_version: 17
    dynamic_axes:  true

monitoring:
  telemetry:
    enabled: false
  metrics:
    log_every_n_steps: 10
  anomaly:
    loss_spike_sigma: 3.0
    grad_norm_sigma:  4.0

data:
  prefetch_factor: 2
  num_workers:     4
EOF

cat > "config/logging.yaml" << 'EOF'
version: 1
disable_existing_loggers: false
formatters:
  json:
    format: '%(message)s'
handlers:
  console:
    class: logging.StreamHandler
    formatter: json
    stream: ext://sys.stdout
  file:
    class: logging.handlers.RotatingFileHandler
    formatter: json
    filename: egx.log
    maxBytes: 52428800
    backupCount: 3
    encoding: utf-8
loggers:
  egx:
    level: INFO
    handlers: [console, file]
    propagate: false
root:
  level: WARNING
  handlers: [console]
EOF

# =============================================================================
# DOCS/
# =============================================================================
mkdir -p "docs/architecture/decisions"
cat > "docs/architecture/README.md" << 'EOF'
# EGX Architecture Docs
- `EGX_Definitive_Architecture.docx` — Complete 13-chapter spec
- `decisions/` — Architecture Decision Records
EOF

cat > "docs/architecture/decisions/ADR-001-dsa-choices.md" << 'EOF'
# ADR-001: DSA Structure Selection
Status: ACCEPTED

| # | Structure | File | Key Operation | Justification |
|---|-----------|------|---------------|---------------|
| 1 | Fibonacci Heap | intelligence/strategy/selector.py | O(1) decrease-key | Live re-score under pressure |
| 2 | Red-Black Tree | intelligence/estimator/calibration/store.py | O(log n) range | Similar-hardware lookup |
| 3 | Skip List | orchestration/pressure/monitor.py | O(log n) lock-free | Concurrent event log |
| 4 | Segment Tree | intelligence/estimator/dryrun.py | O(log n) range-max | Measurement window peaks |
| 5 | Dijkstra | intelligence/graph/topology_graph.py | O((V+E)logV) | Min-latency tensor routing |
| 6 | Kahn's BFS | intelligence/graph/dependency_dag.py | O(V+E) | Import cycle detection |
| 7 | Binary Search | intelligence/strategy/batch_optimizer.py | O(log n) | Monotone VRAM predicate |
| 8 | Trie | runtime/config_loader.py | O(k) | Prefix queries + autocomplete |
EOF

cat > "docs/architecture/decisions/ADR-002-layer-boundaries.md" << 'EOF'
# ADR-002: Layer Import Boundaries
Status: ACCEPTED
Rule: Layer N may ONLY import from layers < N.
Enforced at startup by Kahn's algorithm in dependency_dag.py.

| Layer | Package | Allowed imports |
|-------|---------|-----------------|
| 1 | core/ | stdlib only |
| 2 | infrastructure/ | L1 + torch + pynvml |
| 3 | intelligence/ | L1-2, no torch |
| 4 | orchestration/, resilience/ | L1-3 |
| 5 | training/, peft/, models/, export/, data/, monitoring/ | L1-4 + torch |
| 6 | runtime/ | L1-5 |
| 7 | api/, cli/ | L1-6 |
EOF

cat > "docs/architecture/decisions/ADR-003-models-layer.md" << 'EOF'
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
EOF

cat > "docs/architecture/decisions/ADR-004-export-layer.md" << 'EOF'
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
EOF

# =============================================================================
# TESTS/
# =============================================================================
mkdir -p tests tests/unit tests/unit/core/memory tests/unit/dsa
printf '' > "tests/__init__.py"
mkdir -p tests/unit/intelligence/estimator/calibration
mkdir -p tests/unit/intelligence/planner tests/unit/intelligence/strategy tests/unit/intelligence/graph
mkdir -p tests/unit/orchestration/executor tests/unit/orchestration/pressure tests/unit/orchestration/swapper
mkdir -p tests/unit/resilience/checkpoint
mkdir -p tests/unit/training tests/unit/peft tests/unit/models tests/unit/export
mkdir -p tests/unit/data tests/unit/monitoring tests/unit/runtime
mkdir -p tests/integration tests/gpu_validation tests/benchmarks tests/mocks

cat > "tests/conftest.py" << 'EOF'
# Shared pytest fixtures: MockGPU, MockTopology, MockModelProfile, MockNVML.
# All function-scoped unless explicitly session-scoped.
EOF

# core
mkdir -p "tests/unit/core/memory"
printf '' > "tests/unit/__init__.py"
printf '' > "tests/unit/core/__init__.py"
printf '' > "tests/unit/core/memory/__init__.py"
for f in test_enums test_exceptions test_models test_constants; do
  mkf "tests/unit/core/${f}.py" "Unit: core/${f#test_}.py"
done
for f in test_value test_validators test_units test_formatting; do
  mkf "tests/unit/core/memory/${f}.py" "Unit: core/memory/${f#test_}.py"
done

# DSA — 8 structures, 100% pass required
mkdir -p "tests/unit/dsa"
printf '' > "tests/unit/dsa/__init__.py"
for f in test_fibonacci_heap test_red_black_tree test_skip_list test_segment_tree \
         test_dijkstra test_kahns_bfs test_binary_search test_trie; do
  mkf "tests/unit/dsa/${f}.py" "DSA: correctness + O() complexity proof for ${f#test_}"
done

# intelligence
mkdir -p "tests/unit/intelligence/estimator/calibration"
mkdir -p "tests/unit/intelligence/planner"
mkdir -p "tests/unit/intelligence/strategy"
mkdir -p "tests/unit/intelligence/graph"
for d in "tests/unit/intelligence" \
         "tests/unit/intelligence/estimator" \
         "tests/unit/intelligence/estimator/calibration" \
         "tests/unit/intelligence/planner" \
         "tests/unit/intelligence/strategy" \
         "tests/unit/intelligence/graph"; do
  printf '' > "${d}/__init__.py"
done
for f in test_base test_analytical test_dryrun test_hybrid; do
  mkf "tests/unit/intelligence/estimator/${f}.py" "Unit: estimator/${f#test_}.py"
done
for f in test_cache test_store test_regression; do
  mkf "tests/unit/intelligence/estimator/calibration/${f}.py" "Unit: calibration/${f#test_}.py"
done
for f in test_memory_planner test_allocation_planner test_timeline_planner test_topology_planner; do
  mkf "tests/unit/intelligence/planner/${f}.py" "Unit: planner/${f#test_}.py"
done
for f in test_selector test_scorer test_batch_optimizer test_parallel_advisor; do
  mkf "tests/unit/intelligence/strategy/${f}.py" "Unit: strategy/${f#test_}.py"
done
for f in test_dependency_dag test_topology_graph; do
  mkf "tests/unit/intelligence/graph/${f}.py" "Unit: graph/${f#test_}.py"
done

# orchestration
mkdir -p "tests/unit/orchestration/executor"
mkdir -p "tests/unit/orchestration/pressure"
mkdir -p "tests/unit/orchestration/swapper"
for d in "tests/unit/orchestration" \
         "tests/unit/orchestration/executor" \
         "tests/unit/orchestration/pressure" \
         "tests/unit/orchestration/swapper"; do
  printf '' > "${d}/__init__.py"
done
for sub in executor pressure swapper; do
  mkf "tests/unit/orchestration/${sub}/test_${sub}.py" "Unit: orchestration/${sub}/"
done

# resilience
mkdir -p "tests/unit/resilience/checkpoint"
printf '' > "tests/unit/resilience/__init__.py"
printf '' > "tests/unit/resilience/checkpoint/__init__.py"
mkf "tests/unit/resilience/test_watchdog.py"      "Unit: resilience/watchdog.py — heartbeat, timeout trigger."
mkf "tests/unit/resilience/test_recovery_fsm.py"  "Unit: recovery_fsm.py — all 9 failure types, escalation chain."
mkf "tests/unit/resilience/test_sanitizer.py"     "Unit: sanitizer.py — NaN/Inf detection per tensor."
mkf "tests/unit/resilience/checkpoint/test_manager.py" "Unit: checkpoint/manager.py — strategy selection."
mkf "tests/unit/resilience/checkpoint/test_writer.py"  "Unit: writer.py — atomic write, SHA256, power-loss sim."
mkf "tests/unit/resilience/checkpoint/test_reader.py"  "Unit: reader.py — SHA256 verify, fallback chain, corrupt detect."

# training
mkdir -p "tests/unit/training"
printf '' > "tests/unit/training/__init__.py"
for f in test_kernel test_elastic_loop test_mixed_precision test_gradient_accumulation test_throughput_tracker; do
  mkf "tests/unit/training/${f}.py" "Unit: training/${f#test_}.py"
done

# peft
mkdir -p "tests/unit/peft"
printf '' > "tests/unit/peft/__init__.py"
for f in test_injector test_lora test_qlora test_lora_plus test_dora test_merger; do
  mkf "tests/unit/peft/${f}.py" "Unit: peft/${f#test_}.py"
done

# models [NEW]
mkdir -p "tests/unit/models"
printf '' > "tests/unit/models/__init__.py"
for f in test_registry test_loader test_introspector test_auto_detect; do
  mkf "tests/unit/models/${f}.py" "Unit: models/${f#test_}.py"
done

# export [NEW]
mkdir -p "tests/unit/export"
printf '' > "tests/unit/export/__init__.py"
for f in test_base_exporter test_lora_merger test_safetensors_exporter test_onnx_exporter; do
  mkf "tests/unit/export/${f}.py" "Unit: export/${f#test_}.py"
done

# data
mkdir -p "tests/unit/data"
printf '' > "tests/unit/data/__init__.py"
for f in test_loader test_prefetcher test_streaming test_collator; do
  mkf "tests/unit/data/${f}.py" "Unit: data/${f#test_}.py"
done

# monitoring
mkdir -p "tests/unit/monitoring"
printf '' > "tests/unit/monitoring/__init__.py"
for f in test_metrics test_anomaly_detection test_telemetry; do
  mkf "tests/unit/monitoring/${f}.py" "Unit: monitoring/${f#test_}.py"
done

# runtime
mkdir -p "tests/unit/runtime"
printf '' > "tests/unit/runtime/__init__.py"
for f in test_engine test_lifecycle test_plugin_registry test_config_loader; do
  mkf "tests/unit/runtime/${f}.py" "Unit: runtime/${f#test_}.py"
done

# integration
mkdir -p "tests/integration"
printf '' > "tests/integration/__init__.py"
mkf "tests/integration/test_planning_pipeline.py"   "Integration: probe -> estimate -> strategy -> allocation plan."
mkf "tests/integration/test_recovery_pipeline.py"   "Integration: fault inject -> FSM -> training resumes."
mkf "tests/integration/test_checkpoint_pipeline.py" "Integration: train -> write -> corrupt -> restore."
mkf "tests/integration/test_peft_pipeline.py"       "Integration: model + plan -> inject adapters -> first step."
mkf "tests/integration/test_elastic_batch.py"       "Integration: pressure events -> elastic batch -> effective batch preserved."
mkf "tests/integration/test_zero_config.py"         "Integration: EGX().train() on MockGPU. Zero config. Full flow."
mkf "tests/integration/test_export_pipeline.py"     "Integration [NEW]: train LoRA -> merge -> safetensors -> verify."
mkf "tests/integration/test_model_loading.py"       "Integration [NEW]: load from path -> ModelProfile -> TrainingPlan."

# GPU validation
mkdir -p "tests/gpu_validation"
cat > "tests/gpu_validation/__init__.py" << 'EOF'
# GPU validation. Requires real CUDA GPU. pytest -m gpu
EOF
mkf "tests/gpu_validation/test_dry_run_estimator.py"    "GPU: dryrun vs analytical. Must be within +-10%."
mkf "tests/gpu_validation/test_full_train_loop.py"      "GPU: GPT-2 small 10 steps. OOM=0. Loss decreases."
mkf "tests/gpu_validation/test_oom_recovery.py"         "GPU: deliberate OOM -> batch halve -> training resumes."
mkf "tests/gpu_validation/test_vram_allocation.py"      "GPU: AllocationPlan drift <10% vs post-placement measure."
mkf "tests/gpu_validation/test_nvlink_detection.py"     "GPU: NVLink peer IDs match pynvml P2PStatus."
mkf "tests/gpu_validation/test_lora_merge_accuracy.py"  "GPU [NEW]: merge LoRA -> merged vs original delta <1e-4."
mkf "tests/gpu_validation/test_export_roundtrip.py"     "GPU [NEW]: safetensors export -> reload -> delta <1e-6."

# benchmarks
mkdir -p "tests/benchmarks"
printf '' > "tests/benchmarks/__init__.py"
mkf "tests/benchmarks/test_estimation_speed.py" "Benchmark: analytical <1ms, hybrid <10ms, dryrun <45s."
mkf "tests/benchmarks/test_dsa_throughput.py"   "Benchmark: all 8 DSA under sustained load. Regression baseline."
mkf "tests/benchmarks/test_startup_time.py"     "Benchmark: probe -> plan total <20s on reference hardware."
mkf "tests/benchmarks/test_export_speed.py"     "Benchmark [NEW]: 7B safetensors <60s. LoRA merge <5s."

# mocks
mkdir -p "tests/mocks"
cat > "tests/mocks/__init__.py" << 'EOF'
# MockGPU, MockTopology, MockModelProfile, MockNVML, MockDataLoader.
EOF
mkf "tests/mocks/mock_gpu.py"        "MockGPU: configurable GPUSpec. VRAM, compute_capability, NVLink peers."
mkf "tests/mocks/mock_topology.py"   "MockTopology: LAPTOP/WORKSTATION/DATACENTER/CLUSTER presets."
mkf "tests/mocks/mock_model.py"      "MockModelProfile: configurable param count, hidden_dim, layers."
mkf "tests/mocks/mock_nvml.py"       "MockNVML: pynvml stub. Configurable VRAM used/free per step."
mkf "tests/mocks/mock_dataloader.py" "MockDataLoader: deterministic batches. No real dataset needed."

# =============================================================================
# CI/CD
# =============================================================================
mkdir -p ".github/workflows"
cat > ".github/workflows/ci.yml" << 'EOF'
name: EGX CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  quality:
    name: Code Quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: black --check egx/
      - run: ruff check egx/
      - run: mypy --strict egx/
      - run: python -c "import egx; print('v' + egx.__version__)"

  test-unit:
    name: Unit + DSA Tests
    runs-on: ubuntu-latest
    needs: quality
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit/dsa/ -v --tb=short
        name: DSA tests (100% required)
      - run: pytest tests/unit/ --cov=egx --cov-fail-under=85
        name: All unit tests (>=85% coverage)

  test-integration:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: test-unit
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: pytest tests/integration/ -v

  gpu-validation:
    name: GPU Validation (main only)
    runs-on: [self-hosted, gpu]
    needs: test-integration
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev,flash]"
      - run: pytest tests/gpu_validation/ -v -m gpu
EOF

# =============================================================================
# SUMMARY
# =============================================================================
echo ""

src=$(find "$ROOT" -name '*.py' 2>/dev/null | wc -l | tr -d ' ')
tst=$(find tests -name '*.py' 2>/dev/null | wc -l | tr -d ' ')
cfg=$(find config -type f 2>/dev/null | wc -l | tr -d ' ')
doc=$(find docs .github -type f 2>/dev/null | wc -l | tr -d ' ')
roo=$(ls pyproject.toml README.md .gitignore .python-version install.bat install.sh 2>/dev/null | wc -l | tr -d ' ')

echo "  ┌─────────────────────────────────────────┐"
echo "  │  EGX v2 Bootstrap Complete               │"
echo "  ├─────────────────────────────────────────┤"
printf "  │  Source files  (egx/)      %4s files    │\n" "$src"
printf "  │  Test files    (tests/)    %4s files    │\n" "$tst"
printf "  │  Config files  (config/)   %4s files    │\n" "$cfg"
printf "  │  Docs + CI     (docs/.gh/) %4s files    │\n" "$doc"
printf "  │  Root files                %4s files    │\n" "$roo"
echo "  ├─────────────────────────────────────────┤"
printf "  │  TOTAL                     %4s files    │\n" "$((src+tst+cfg+doc+roo))"
echo "  ├─────────────────────────────────────────┤"
echo "  │  Changes from v1:                        │"
echo "  │  + egx/models/      (from old repo)      │"
echo "  │  + egx/export/      (from old repo)      │"
echo "  │  + egx/peft/merger.py (deploy critical)  │"
echo "  │  + install.bat      (Windows support)    │"
echo "  │  + install.sh       (Linux/macOS)        │"
echo "  │  ✓ Fixed __init__.py Law 4 violation     │"
echo "  │  + 12 new test files (models + export)   │"
echo "  │  + 4 ADRs (decisions documented)         │"
echo "  └─────────────────────────────────────────┘"
echo ""
echo "  Next steps:"
echo "  Windows: run install.bat"
echo "  Linux:   bash install.sh"
echo "  Then:    pytest tests/unit/  (should collect, no failures)"