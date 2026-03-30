# EGX Codebase Navigation Guide
## Quick Reference for Developers

---

## 1. Directory Structure & Responsibilities

### Core Foundation (Layer 1)
```

egx/core/
├─ interfaces.py         → ABC contracts for all extension points
├─ exceptions.py         → Hierarchical error system with recovery hints  
├─ models.py            → Frozen dataclasses for all data contracts
├─ enums.py             → Type-safe enums (string-based for JSON)
├─ constants.py         → System-wide constants (no magic numbers elsewhere!)
├─ device.py            → Device detection (CUDA/MPS/CPU)
└─ memory/
   ├─ units.py          → KB, MB, GB, TB constants
   ├─ value.py          → MemoryValue wrapper (prevents unit confusion)
   └─ validators.py     → Check memory bounds
```

**Key Insight:** Everything at this layer is immutable & frozen. No side effects.

### Infrastructure Layer (Layer 2)
```
egx/infrastructure/
├─ gpu_probe.py         → Hardware enumeration (NVML-based)
├─ topology_builder.py  → Assemble GPUs into unified topology
├─ bandwidth_sampler.py → Measure interconnect & NVMe speeds
│                          (Uses Skip List DSA pattern)
├─ gpu_spec_cache.py    → Memoize GPU queries (expensive!)
└─ thermal_monitor.py   → Monitor GPU temperatures
```

**Entry Point:** `GPUProber().probe()` → `List[GPUSpec]`  
**Key Pattern:** Context manager (RAII) for NVML lifecycle

### Intelligence Layer (Layer 3)
```
egx/intelligence/
├─ estimator/
│  ├─ base.py               → `estimate(topology, profile, plan) -> MemoryReport`
│  ├─ analytical.py         → Formula-based (fast, <1ms, ±15% error)
│  ├─ hybrid.py             → Blend multiple estimators
│  ├─ ml_based.py           → Pretrained model (planned)
│  └─ calibrator.py         → Learn correction factors per hardware
├─ strategy/
│  ├─ scorer.py             → Score strategies against situation
│  └─ selector.py           → Pick best strategy (Fibonacci heap)
├─ planner/
│  └─ adaptive_batch.py     → Find optimal batch size via binary search
└─ optimizer/
   └─ graph_optimizer.py    → Optimize training DAG
```

**Critical Function:** `StrategyScorer.score_all()` → ranked list of strategies  
**Flow:** Hardware info → Memory estimate → Strategy scores → Best fit

### Resilience Layer (Layer 4)
```
egx/resilience/
├─ sanitizer.py         → Pre-flight checks (NaN/Inf detection)
├─ watchdog.py          → Deadlock detection with heartbeats
├─ recovery/
│  ├─ manager.py        → Orchestrate recovery attempts
│  └─ strategies.py     → Individual recovery tactics (Retry, HalveBatch, etc)
├─ checkpoint/
│  ├─ manager.py        → Adaptive checkpoint scheduling
│  └─ selector.py       → Which checkpoint to keep?
└─ telemetry.py         → Anomaly detection (loss spikes, etc)
```

**Key Class:** `RecoveryOrchestrator` (planned: coordinates recovery chain)  
**Current Issue:** Recovery not yet fully wired up

### Training Layer (Layer 5) — LARGEST LAYER
```
egx/training/
├─ kernel.py                    → Execute single training step ⭐
├─ gradient_accumulation.py     → Batching gradient updates
├─ data/
│  ├─ loader.py                 → Hardware-aware DataLoader
│  ├─ collator.py               → Custom batch assembly
│  ├─ prefetcher.py             → Async data prefetch
│  └─ streaming.py              → Infinite streaming datasets
├─ adapters/
│  ├─ lora.py                   → LoRA implementation ⭐
│  ├─ dora.py                   → DoRA (planned)
│  ├─ qlora.py                  → 4-bit quantized LoRA
│  └─ merger.py                 → Merge adapters back to base
├─ models/
│  ├─ loader.py                 → Load transformers safely
│  ├─ introspector.py           → Analyze model architecture
│  └─ registry.py               → Model zoo
└─ export/
   ├─ base_exporter.py          → Export interface
   ├─ safetensors_exporter.py   → Sharded SafeTensors format
   └─ onnx_exporter.py          → ONNX for inference
```

**⭐ Critical Files:**
- `kernel.py`: Where loss.backward() happens
- `lora.py`: Where parameter efficiency magic occurs

### Orchestration Layer (Layer 6)
```
egx/orchestration/
├─ engine.py            → 10-phase lifecycle manager ⭐⭐
├─ executor.py          → Execute individual phases
├─ memory/
│  └─ budget.py         → Per-GPU memory allocator
├─ pressure/
│  └─ monitor.py        → Monitor memory/thermal pressure levels
└─ swapper.py           → Offload to NVMe under pressure
```

**⭐⭐ Most Important:** `EGXEngine.run_training()` orchestrates the entire workflow  
**10 Phases:**
1. Hardware probing
2. Topology assembly
3. Model loading
4. Strategy selection
5. Kernel initialization
6. Data loading
7. Training loop setup
8. Training execution
9. Evaluation
10. Cleanup

### API Layer (Layer 7) — USER FACING
```
egx/api/
├─ trainer.py           → `EGXTrainer` main entry point ⭐⭐⭐
├─ config.py            → `EGXConfig` with validation & defaults
├─ callbacks.py         → Lifecycle hooks (TrainingCallback base class)
├─ evaluator.py         → Standalone evaluation
├─ predictor.py         → Inference / generation
├─ validation.py        → Safety checks before training
└─ cli/
   └─ main.py           → `egx` CLI commands (probe, train, benchmark)
```

**⭐⭐⭐ Most Important:** This is what users touch  
```python
trainer = EGXTrainer()  # Zero-config!
result = trainer.train(model, dataset)  # "Just works"
```

### Plugins (Optional Extensions)
```
egx/plugins/
├─ cpu_offload.py       → Offload to CPU RAM
├─ flash_attention.py   → Flash Attention v2 optimizer
├─ gradient_checkpointing.py → Activation checkpointing
└─ zero3.py             → DeepSpeed ZeRO Stage 3
```

---

## 2. Critical Paths (Trace a Training Run)

### Path A: Initialize Trainer
```
User Code
  ↓
EGXTrainer.__init__()
  ├─ EGXConfig() validation
  ├─ EGXEngine() initialization
  ├─ CallbackHandler setup
  └─ LoggingCallback auto-add
```

### Path B: Boot Trainer (First Time Only)
```
trainer.train(model, dataset)
  ↓
EGXEngine.boot(model, config)
  ├─ GPUProber.probe() with RAII context
  │  └─ NVML init → query GPUs → NVML shutdown
  ├─ TopologyBuilder.build(gpus)
  │  └─ Assemble into HardwareTopology
  ├─ ModelValidator.check_nans(model)
  └─ Log: "Boot successful"
```

### Path C: Strategy Selection
```
EGXEngine.run_training()
  ↓ Phase 5:
StrategyScorer.score_all(gpu, model_bytes, [ALL_MODES])
  ├─ For each mode:
  │  ├─ AnalyticalEstimator.estimate() → MemoryReport
  │  ├─ Score: (safety * 0.4 + speed * 0.25 + peff * 0.20 + user * 0.15)
  │  └─ Store: (score, mode)
  └─ Return: sorted strategies
  ↓
  Best strategy selected & mode set
```

### Path D: Training Loop (Main)
```
EGXEngine.run_training()
  ↓ Phases 6-8:
for epoch in range(num_epochs):
  for batch in dataloader:
    ├─ InputSanitizer.check_batch() → catch NaN/Inf early
    ├─ TrainingKernel.train_step(batch, step)
    │  ├─ Forward: loss = model(batch)
    │  ├─ Backward: loss.backward()
    │  ├─ Clip: clip_grad_norm_()
    │  ├─ Step: optimizer.step()
    │  └─ Return: loss.item()
    ├─ Callback: on_step_end(loss=loss, lr=lr)
    ├─ CheckpointManager: should_save(step, loss)?
    │  └─ If yes: save checkpoint + cleanup old ones
    └─ Watchdog.heartbeat(step) → deadlock detection
```

### Path E: Recovery (If Error)
```
try:
  train_step()
except OutOfMemoryError as e:
  ├─ context = RecoveryContext(error=e, step=step, ...)
  ├─ RecoveryOrchestrator.recover(context)
  │  ├─ Strategy 0: RetryStrategy.attempt()
  │  ├─ Strategy 1: HalveBatchStrategy.attempt()
  │  ├─ Strategy 2: DowngradeStrategyStrategy.attempt()
  │  └─ Strategy 3: CheckpointRollbackStrategy.attempt()
  └─ If all fail: raise EGXError(recoverable=False)
```

---

## 3. Key Design Patterns

### Pattern 1: Dependency Injection
```python
# Bad ❌
trainer = EGXTrainer()
trainer._gpu_prober = GPUProber()  # Hidden dependency

# Good ✅
trainer = EGXTrainer(
    gpu_prober=GPUProber(),  # Explicit
    topology_builder=TopologyBuilder(),
)
```

### Pattern 2: Context Manager (RAII)
```python
# Bad ❌
nvml.nvmlInit()
try:
    gpus = probe_gpus()
finally:
    nvml.nvmlShutdown()

# Good ✅
with GPUProber() as prober:
    gpus = prober.probe()  # Auto cleanup
```

### Pattern 3: Strategy Pattern
```python
# Different training modes, same interface
strategies = [
    FullFinetune(),
    LoRA(),
    QLoRA(),
]

selected = select_best(strategies)  # Based on hardware
model = selected.setup(model)  # Setup adapters
loss = selected.train_step()   # Train
```

### Pattern 4: Observer (Callbacks)
```python
trainer = EGXTrainer(callbacks=[
    LoggingCallback(),
    EarlyStoppingCallback(),
    SaveBestCallback(),
])

# Trainer fires hooks at each phase
trainer._callback_handler.fire("on_step_end", step=100, loss=0.5)
```

### Pattern 5: Factory Pattern
```python
# Create right instance for situation
estimator = get_estimator(config.estimator_type)
# Returns: AnalyticalEstimator | HybridEstimator | MLBasedEstimator

model = ModelLoader().load(model_id, device="cuda")
# Returns: HF model | Mock model | Custom model
```

---

## 4. Common Tasks & Where to Look

### "I want to add a new training mode"
1. Add enum in `egx/core/enums.py`: `TrainingMode.MY_MODE = "my_mode"`
2. Implement injector in `egx/training/adapters/my_mode.py`
3. Register in `StrategyScorer.score_all()` to score it
4. Update memory formula in `AnalyticalEstimator.estimate()`
5. Add tests in `tests/unit/test_strategies.py`

### "I want to improve memory estimation accuracy"
1. Check `egx/intelligence/estimator/analytical.py` (likely issue with `ACTIVATION_FACTOR_DEFAULT`)
2. Measure actual peak memory via dry-run (see `MLBasedEstimator` planned code)
3. Add test in `tests/unit/test_estimators.py` with `test_memory_estimate_vs_actual`
4. Consider ML model approach in `egx/intelligence/estimator/ml_based.py`

### "I want to add a new hardware backend (AMD, TPU, etc)"
1. Create `egx/infrastructure/probers/amd_prober.py` (inherit `BaseGPUProber`)
2. Implement device detection & spec query
3. Update `GPUProber.probe()` to try AMD fallback after NVIDIA
4. Add topology assembly logic for interconnects
5. Test in `tests/gpu_validation/test_amd_support.py`

### "I want to improve training throughput"
1. Profile with: `python -m cProfile -s cumtime train.py`
2. Check `TrainingKernel.train_step()` - is gradient accumulation working?
3. Check data loading: `NVMeDataLoader` should prefetch → see `egx/data/loader.py`
4. Check for unnecessary allocations: `torch.cuda.memory_reserved()` spikes?
5. Current bottleneck likely: **activation memory recomputation**
   → Enable gradient checkpointing in `EGXConfig(gradient_checkpointing=True)`

### "I want to add monitoring/logging"
1. Add metric in `egx/monitoring/metrics.py`
2. Create callback that tracks it: inherit `TrainingCallback`
3. Example: `LoggingCallback` in `egx/api/callbacks.py`
4. For Prometheus export: see planned `egx/monitoring/prometheus_exporter.py`

### "I want to support distributed training"
1. Planned in roadmap: `egx/api/distributed.py`
2. Create `DistributedTrainer(EGXTrainer)` subclass
3. Wrap model with `DDP` or `FSDP` based on config
4. Shard dataset with `DistributedSampler`
5. Handle gradient synchronization automatically
6. Test with: `torchrun --nproc_per_node=8 train.py`

---

## 5. Common Pitfalls & Solutions

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| **OOM at batch size that should fit** | Activation memory overestimate too aggressive | Set `gradient_checkpointing=False` to see real usage |
| **Loss becomes NaN suddenly** | Gradient explosion or data corruption | Enable `InputSanitizer(strict=True)` |
| **Training freeze (no error)** | Deadlock in data loading | Check watchdog timeout; increase HEARTBEAT_INTERVAL_S |
| **Strategy selection always chooses LoRA** | Memory estimator too conservative | Run calibration: use ML-based estimator |
| **Checkpoint saving every second** | CheckpointManager too aggressive | Increase `checkpoint_strategy.step_interval` |
| **Memory grows over 1000 steps** | GPU memory leak in model code | Profile with `torch.cuda.max_memory_allocated()` |
| **"No compatible GPUs detected"** | NVML initialization failed | Check NVIDIA drivers; fallback to CPU mode works |

---

## 6. Testing Strategy

### Unit Tests (Fast, Isolated)
```bash
pytest tests/unit/               # <5 min, no GPU needed
# Tests core logic: estimators, scorers, exceptions
```

### Integration Tests (Slow, Real Models)
```bash
pytest tests/integration/        # 10-30 min, may use GPU
# Tests end-to-end: boot → train → eval → export
```

### Performance Tests
```bash
pytest tests/performance/        # 1+ hours
# Tests throughput, memory, regression
```

### GPU Validation (Hardware-Specific)
```bash
pytest tests/gpu_validation/     # GPU-specific, varies by hardware
# Tests NVML, topology detection, actual memory
```

---

## 7. Configuration Defaults (Zero-Config Design)

Every config has sensible defaults. Users typically only override:

```python
# Minimal config
config = EGXConfig(
    learning_rate=1e-5,      # Only thing that usually needs tweaking
    batch_size=4,            # Hardware-dependent
)

# Everything else auto-tunes:
# - Training mode (adaptive)
# - Optimizer (AdamW)
# - Scheduler (None, linear, or cosine)
# - Checkpointing (adaptive)
# - Callbacks (auto-add logging + early stopping)
```

---

## 8. Performance Optimization Checklist

When things are too slow:

- [ ] Is gradient accumulation actually accumulating? Check `TrainingKernel.train_step()` implementation
- [ ] Are gradients being accumulated with proper scaling (1/num_accumulation_steps)?
- [ ] Is `num_workers` in DataLoader set correctly? (Should be ~GPU_count * 2)
- [ ] Is `prefetch_factor` sized for NVMe speed? (See `NVMeDataLoader`)
- [ ] Is `pin_memory=True` set in DataLoader?
- [ ] Is mixed precision enabled? (`precision_override="bf16"`)
- [ ] Is gradient checkpointing enabled? (Reduce memory, increase compute)
- [ ] Is batch size optimal? (Use `AdaptiveBatchSearcher`)
- [ ] Are there NaN/Inf that cause recovery slowdown? (Check `InputSanitizer`)

---

## 9. Useful Commands

```bash
# Probe hardware
egx probe

# Training with config file
egx train --config myconfig.yaml

# Benchmark memory
egx benchmark --model llama-7b --batch-size 4

# Python API
python
>>> from egx.api.trainer import EGXTrainer
>>> trainer = EGXTrainer()
>>> result = trainer.train(model, dataset)

# Run tests with coverage
pytest --cov=egx tests/unit/ --cov-report=html

# Type checking
mypy --strict egx/

# Linting
ruff check egx/ tests/
```

---

## 10. Where to Find Documentation

- **README.md**: Overview & quickstart
- **SENIOR_CODE_REVIEW.md**: Comprehensive assessment
- **UPGRADE_ROADMAP.md**: Planned improvements with code
- **This file**: Navigation & reference
- **Docstrings**: In-code documentation for all public APIs
- **Type hints**: Read `.pyi` stubs for function contracts
- **Config defaults**: `EGXConfig` dataclass in `egx/api/config.py`

---

## Final Note

**OG Project Organization:** EGX is well-structured but getting complex. If confused:

1. **Find the layer** (1-7) that handles your problem
2. **Read that layer's README** (if it exists)
3. **Check base interfaces** in `egx/core/interfaces.py`
4. **Look for tests** in `tests/` for usage examples
5. **Ask:** "What's the simplest, most testable approach?"

**Happy coding! 🚀**

