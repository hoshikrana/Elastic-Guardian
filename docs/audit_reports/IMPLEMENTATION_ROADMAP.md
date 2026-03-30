# EGX Framework: Implementation Roadmap
**Based on Comprehensive Senior Review | March 27, 2026**

---

## Overview

This document consolidates findings from **Phase 1-3 reviews** (Architecture 8.5/10, Code Quality 6.8/10, Performance 6.8/10) and outlines a **7-week implementationroadmap** to achieve **9.0/10 overall quality**.

**Current Status:** 7.4/10 (Production-Ready)  
**Target Status:** 9.0/10 (Polished & Optimized)  
**Total Effort:** ~42 hours (5-6 developer-weeks)

---

## Implementation Priorities

### 🔴 Priority 0: Critical (Foundation) — 10 hours
Must fix before rolling to production at scale

| Task | Issue | Solution | Effort | Status |
|------|-------|----------|--------|--------|
| **Refactor `_production_training_loop()`** | 400+ lines, 18+ CC | Split into 5 methods | 8h | ⏳ Planned |
| **Normalize `selected_mode` type** | String vs enum ambiguity | Always use enum | 2h | ⏳ Planned |

---

### 🟠 Priority 1: High Impact — 15 hours
Significant improvements in maintainability/performance

| Task | Issue | Solution | Effort | Status |
|------|-------|----------|--------|--------|
| **Extract `TrainingSessionConfig`** | 15+ getattr duplications | Create config class | 2.5h | ⏳ Planned |
| **Strategy pattern for loss functions** | String matching on errors | Abstract loss calc | 4h | ⏳ Planned |
| **Empirical profiling** | Memory estimates unvalidated | Add runtime tracking | 4h | ⏳ Planned |
| **Optimize NVMe swapper** | torch.save/load slow | SafeTensors + async | 6h | ⏳ Planned |

---

### 🟡 Priority 2: Polish — 15 hours
Refinements for production excellence

| Task | Issue | Solution | Effort | Status |
|------|-------|----------|--------|--------|
| **Integrate plugin system** | Plugins not usable | Add CLI flags | 3h | ⏳ Planned |
| **Adaptive DataLoader tuning** | Heuristics untested | Profile batch latency | 6h | ⏳ Planned |
| **Backend strategy decision** | Abstraction incomplete | Implement or remove | 4h | ⏳ Planned |
| **Add callback TypedDict** | Context undocumented | Document **kwargs | 2h | ⏳ Planned |

---

## Detailed Implementation Plan

---

## PHASE IMPLEMENTATION: P0 — Foundation (Critical)

### Task P0.1: Refactor `_production_training_loop()` ⏳ [8 hours]

**Location:** [egx/runtime/engine.py](egx/runtime/engine.py#L244)

**Current State:**
- 400+ lines, 18+ cyclomatic complexity
- 8 responsibilities mixed together
- Nested conditionals, multiple break-from-nested-loops

**Target State:**
Extract into 5 focused methods:

```python
# egx/runtime/engine.py

def _production_training_loop(self, ...) -> Dict[str, Any]:
    """Orchestrator for all training phases (simplified)."""
    setup = self._setup_training()              # Phase 5-8: Setup
    best_eval_loss = float('inf')
    
    for epoch in range(setup.epochs):
        metrics = self._run_training_epoch(epoch, setup)
        best_eval_loss = self._maybe_evaluate_and_checkpoint(
            epoch, metrics, best_eval_loss, setup
        )
        if self._should_stop_training(metrics):
            logger.info(f"Stopping training at epoch {epoch}")
            break
    
    return self._finalize_training(setup, best_eval_loss)

def _setup_training(self) -> TrainingSetup:
    """Phases 5-8: Strategy selection, PEFT injection, kernel setup."""
    # ~50 lines: strategy selection, kernel init
    ...

def _run_training_epoch(self, epoch: int, setup: TrainingSetup) -> EpochMetrics:
    """Phase 9: Epoch loop with batches and error recovery."""
    # ~100 lines: batch loop, train_step calls, recovery
    ...

def _maybe_evaluate_and_checkpoint(
    self, epoch: int, metrics: EpochMetrics, 
    best_eval_loss: float, setup: TrainingSetup
) -> float:
    """Phase 10: Post-epoch evaluation and checkpointing."""
    # ~40 lines: eval, checkpoint logic
    ...

def _should_stop_training(self, metrics: EpochMetrics) -> bool:
    """Check early stopping conditions."""
    # ~10 lines
    ...

def _finalize_training(self, setup: TrainingSetup, best_eval_loss: float) -> Dict:
    """Cleanup and final logging."""
    # ~15 lines
    ...
```

**Implementation Steps:**
1. [ ] Create TrainingSetup dataclass (holds config, model, loaders, kernel, etc.)
2. [ ] Extract _setup_training() — consolidate all Phase 5-8 logic
3. [ ] Extract _run_training_epoch() — main epoch loop
4. [ ] Extract _maybe_evaluate_and_checkpoint() — eval + checkpoint
5. [ ] Extract helper methods for error handling
6. [ ] Update _production_training_loop() to orchestrate
7. [ ] Add unit tests for each method
8. [ ] Verify end-to-end training works

**Tests to Add:**
- Unit test _setup_training() with mock config
- Unit test _run_training_epoch() with mock data
- Integration test full cycle

---

### Task P0.2: Normalize `selected_mode` Type ⏳ [2 hours]

**Location:** [egx/runtime/engine.py](egx/runtime/engine.py#L150-180)

**Current Problem:**
```python
# Sometimes string:
selected_mode = "full_finetune"
isinstance(selected_mode, str)  # True

# Sometimes enum:
selected_mode = TrainingMode.LORA
selected_mode.value  # "lora"

# Runtime polymorphism:
if isinstance(selected_mode, str):
    mode_enum = TrainingMode(selected_mode)
else:
    mode_enum = selected_mode
```

**Solution:**
Always normalize to enum at config construction time.

**Implementation:**
1. [ ] Update `run_training()` to always return `TrainingMode` enum
2. [ ] Add guard in strategy selection that guarantees enum
3. [ ] Remove isinstance checks for string
4. [ ] Update type hints: `selected_mode: TrainingMode` (not Union)
5. [ ] Add tests for mode normalization

---

## PHASE IMPLEMENTATION: P1 — High Impact (15 hours)

### Task P1.1: Extract `TrainingSessionConfig` ⏳ [2.5 hours]

**Location:** [egx/api/config.py](egx/api/config.py) — add new class

**Current Problem:**
15+ getattr() calls scattered through run_training() and train_step():
```python
# In run_training():
optimizer_type=getattr(config, "optimizer_type", "adamw")
loss_fn=getattr(config, "loss_fn", None)
learning_rate=getattr(config, "learning_rate", 2e-5)
scheduler_type=getattr(config, "scheduler_type", None)
warmup_steps=getattr(config, "warmup_steps", 0)
callbacks=getattr(config, "callbacks", [])
# ... 12 more fields
```

**Solution:** Extract defaults into single TrainingSessionConfig class

**Implementation:**
```python
# egx/api/config.py

@dataclass(frozen=True)
class TrainingSessionConfig:
    """Runtime training configuration extracted from user config."""
    
    # Strategy & PEFT
    training_mode: TrainingMode = TrainingMode.LORA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_targets: Optional[List[str]] = None
    
    # Optimization
    optimizer_type: str = "adamw"
    learning_rate: float = 2e-5
    scheduler_type: Optional[str] = None
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    
    # Gradient control
    gradient_accumulation_steps: int = 1
    loss_fn: Optional[Union[Callable, str]] = None
    precision_override: Optional[str] = None
    
    # Checkpointing & Recovery
    output_dir: str = "./egx_output"
    checkpoint_strategy: str = "adaptive"
    timeout: float = 300.0
    
    # Callbacks & Hooks
    callbacks: List[TrainingCallback] = field(default_factory=list)
    
    @classmethod
    def from_user_config(cls, config: EGXConfig) -> "TrainingSessionConfig":
        """Extract with defaults, validating types."""
        return cls(
            training_mode=cls._normalize_mode(getattr(config, "training_mode", TrainingMode.LORA)),
            lora_rank=getattr(config, "lora_rank", 16),
            # ... etc
        )
    
    @staticmethod
    def _normalize_mode(mode) -> TrainingMode:
        if isinstance(mode, TrainingMode):
            return mode
        if isinstance(mode, str):
            return TrainingMode(mode)
        raise ValueError(f"Invalid training mode: {mode}")
```

**Implementation Steps:**
1. [ ] Create TrainingSessionConfig dataclass
2. [ ] Add from_user_config() classmethod
3. [ ] Update run_training() to create instance
4. [ ] Remove getattr() calls, use config.field instead
5. [ ] Add tests for config extraction
6. [ ] Update type hints in related functions

---

### Task P1.2: Strategy Pattern for Loss Functions ⏳ [4 hours]

**Location:** [egx/training/kernel.py](egx/training/kernel.py#L140-180)

**Current Problem:**
```python
if callable(self.loss_fn):
    try:
        loss = self.loss_fn(outputs)
    except Exception:
        loss = outputs.loss if hasattr(outputs, "loss") else outputs.sum()
elif isinstance(self.loss_fn, str) and self.loss_fn.lower() == "mse":
    if "labels" not in batch:
        raise ValueError("MSE loss requires 'labels' in batch")
    loss = torch.nn.functional.mse_loss(outputs, batch["labels"])
elif isinstance(self.loss_fn, str) and self.loss_fn.lower() in ["cross_entropy", "ce"]:
    if "labels" not in batch:
        raise ValueError("Cross-Entropy loss requires 'labels' in batch")
    loss = torch.nn.functional.cross_entropy(outputs, batch["labels"])
else:
    loss = outputs.loss if hasattr(outputs, "loss") else outputs.sum()
```

**Solution:** Abstract into strategy classes

```python
# egx/training/loss_strategies.py

class LossFunctionStrategy(ABC):
    """Abstract loss function calculator."""
    
    @abstractmethod
    def compute(self, outputs: Any, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss from model outputs and batch."""
        pass

class CallableLossStrategy(LossFunctionStrategy):
    def __init__(self, fn: Callable):
        self.fn = fn
    
    def compute(self, outputs, batch) -> torch.Tensor:
        try:
            return self.fn(outputs)
        except TypeError:
            # Fallback to outputs.loss or sum
            return outputs.loss if hasattr(outputs, "loss") else outputs.sum()

class MSELossStrategy(LossFunctionStrategy):
    def compute(self, outputs, batch) -> torch.Tensor:
        if "labels" not in batch:
            raise ValueError("MSE requires 'labels' in batch")
        return F.mse_loss(outputs, batch["labels"])

class CrossEntropyStrategy(LossFunctionStrategy):
    def compute(self, outputs, batch) -> torch.Tensor:
        if "labels" not in batch:
            raise ValueError("CrossEntropy requires 'labels' in batch")
        return F.cross_entropy(outputs, batch["labels"])

# Factory
class LossFunctionFactory:
    @staticmethod
    def create(loss_fn) -> LossFunctionStrategy:
        if loss_fn is None:
            return DefaultLossStrategy()
        if callable(loss_fn):
            return CallableLossStrategy(loss_fn)
        if isinstance(loss_fn, str):
            if loss_fn.lower() == "mse":
                return MSELossStrategy()
            elif loss_fn.lower() in ["cross_entropy", "ce"]:
                return CrossEntropyStrategy()
        raise ValueError(f"Unknown loss function: {loss_fn}")
```

**In TrainingKernel:**
```python
class TrainingKernel:
    def __init__(self, ..., loss_fn=None, ...):
        self.loss_strategy = LossFunctionFactory.create(loss_fn)
    
    def train_step(self, batch, step, ...):
        # ... forward pass ...
        loss = self.loss_strategy.compute(outputs, batch)
        # ... backward ...
```

---

### Task P1.3: Add Empirical Profiling ⏳ [4 hours]

**Location:** [egx/training/kernel.py](egx/training/kernel.py) + New file [egx/monitoring/profiler.py](egx/monitoring/profiler.py)

**Goal:** Compare estimated vs. actual memory usage during training

**Implementation:**

```python
# egx/monitoring/profiler.py

@dataclass
class MemorySnapshot:
    allocated_bytes: int
    reserved_bytes: int
    timestamp: float
    phase: str  # "forward", "backward", "optimizer_step"

class MemoryProfiler:
    """Track actual GPU memory usage during training."""
    
    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self.start_time = time.time()
    
    def capture(self, phase: str):
        """Capture memory at specific phase."""
        torch.cuda.synchronize()
        self.snapshots.append(MemorySnapshot(
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
            timestamp=time.time() - self.start_time,
            phase=phase,
        ))
    
    def peak_memory_mb(self) -> float:
        """Return peak memory in MB."""
        if not self.snapshots:
            return 0.0
        peak = max(s.allocated_bytes for s in self.snapshots)
        return peak / (1024 ** 2)
    
    def report(self, estimated_mb: float) -> Dict[str, Any]:
        """Generate profiling report."""
        actual = self.peak_memory_mb()
        error_pct = abs(actual - estimated_mb) / estimated_mb * 100
        
        return {
            "estimated_mb": estimated_mb,
            "actual_mb": actual,
            "error_percent": error_pct,
            "accuracy": "within 9%" if error_pct <= 9 else "NEEDS TUNING",
        }
```

**Integration in TrainingKernel:**
```python
class TrainingKernel:
    def __init__(self, ..., enable_profiling=False, ...):
        self.profiler = MemoryProfiler() if enable_profiling else None
    
    def train_step(self, batch, ...):
        if self.profiler:
            self.profiler.capture("batch_prepare")
        
        outputs = self.model(**batch)
        
        if self.profiler:
            self.profiler.capture("forward_pass")
        
        loss.backward()
        
        if self.profiler:
            self.profiler.capture("backward_pass")
        
        self.optimizer.step()
        
        if self.profiler:
            self.profiler.capture("optimizer_step")
```

---

### Task P1.4: Optimize NVMe Swapper -> SafeTensors + Async Prefetch ⏳ [6 hours]

**Location:** [egx/orchestration/swapper/ram_to_nvme.py](egx/orchestration/swapper/ram_to_nvme.py)

**Current Issue:** torch.save/load serialization is slow (50-500ms per GB)

**Solution:** Use SafeTensors format + async background restore

```python
# egx/orchestration/swapper/ram_to_nvme.py (updated)

import asyncio
import threading
from pathlib import Path
from safetensors.torch import save_file, load_file

class OptimizedNVMeSwapper:
    """Fast tensor swap to NVMe with prefetching."""
    
    def __init__(self, cache_dir: str, prefetch_queue_size: int = 5):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._manifest: Dict[str, Path] = {}
        self._prefetch_queue: asyncio.Queue = asyncio.Queue(maxsize=prefetch_queue_size)
        self._prefetch_cache: Dict[str, torch.Tensor] = {}
        self._lock = threading.RLock()
    
    def offload(self, name: str, tensor: torch.Tensor) -> int:
        """Offload tensor to NVMe using SafeTensors (2-3x faster)."""
        file_path = self._cache_dir / f"{name.replace('.', '_')}.safetensors"
        
        # Use SafeTensors instead of torch.save
        save_file({"tensor": tensor}, str(file_path))
        
        with self._lock:
            self._manifest[name] = file_path
        
        return tensor.nelement() * tensor.element_size()
    
    def restore(self, name: str, device: str = "cpu", prefetch: bool = True) -> torch.Tensor:
        """Load tensor from NVMe (with optional async prefetch of next)."""
        # Check prefetch cache first
        with self._lock:
            if name in self._prefetch_cache:
                return self._prefetch_cache.pop(name)
            
            if name not in self._manifest:
                raise KeyError(f"Tensor '{name}' not found in NVMe cache")
            
            file_path = self._manifest[name]
        
        # Load with SafeTensors
        loaded = load_file(str(file_path), device=device)
        tensor = loaded["tensor"]
        
        return tensor
    
    def prefetch_async(self, names: List[str], device: str = "cpu"):
        """Background prefetch list of tensors."""
        def _prefetch_worker():
            for name in names:
                try:
                    tensor = self.restore(name, device, prefetch=False)
                    with self._lock:
                        self._prefetch_cache[name] = tensor
                except Exception as e:
                    logger.warning(f"Prefetch failed for {name}: {e}")
        
        # Run in background thread
        thread = threading.Thread(target=_prefetch_worker, daemon=True)
        thread.start()
```

---

## PHASE IMPLEMENTATION: P2 — Polish (15 hours)

### Task P2.1: Integrate Plugin System ⏳ [3 hours]

Add user-accessible CLI flags to enable plugins

```bash
# Users should be able to do:
egx train --model llama-7b --dataset data.csv --plugins flash_attn,gradient_checkpoint
```

**Implementation:**
1. [ ] Add `--plugins` flag to [egx/cli/main.py](egx/cli/main.py)
2. [ ] Parse plugin names, validate against available plugins
3. [ ] Load plugins before training starts
4. [ ] Update README with plugin usage examples

---

### Task P2.2: Adaptive DataLoader Tuning ⏳ [6 hours]

Profile batch preparation latency and adjust num_workers/prefetch dynamically

---

### Task P2.3: Backend Strategy Decision ⏳ [4 hours]

Either:
- Remove backend abstraction (simplify to PyTorch-only)
- Implement JAX proof-of-concept

**Recommendation:** Remove for now (clean up tech debt)

---

### Task P2.4: Callback Context TypedDict ⏳ [2 hours]

Document callback **kwargs properly using TypedDict

---

## Testing Strategy

Each implementation task includes tests:

### P0 Tests (16 hours total)
- Unit tests for refactored methods (4h)
- Integration tests for training loop (4h)
- Type safety tests (2h)
- Regression tests (ongoing)

### P1 Tests (12 hours total)
- Config extraction tests (2h)
- Loss strategy unit tests (3h)
- Profiling tests (3h)
- Swapper performance tests (4h)

### P2 Tests (8 hours total)
- Plugin loading tests (2h)
- DataLoader tuning tests (4h)
- End-to-end verification (2h)

---

## Quality Gates

Before marking each task complete:

- [ ] All new code has type hints (100%)
- [ ] All public APIs have docstrings
- [ ] Unit test coverage >90% for new code
- [ ] No regression in existing tests
- [ ] Code review approved by another dev
- [ ] Performance benchmarks measured (if applicable)
- [ ] No new complexity hotspots introduced

---

## Timeline

**Week 1 (P0 — Foundation):**
- Complete _production_training_loop() refactor (8h)
- Complete selected_mode normalization (2h)
- Testing & validation (6h)
- **Subtotal: 16 hours (2 FTE-days)**

**Week 2-3 (P1 — High Impact):**
- TrainingSessionConfig extraction (2.5h)
- Loss function strategy pattern (4h)
- Empirical profiling (4h)
- NVMe swapper optimization (6h)
- Testing & integration (8h)
- **Subtotal: 24.5 hours (3 FTE-days)**

**Week 4-5 (P2 — Polish):**
- Plugin system integration (3h)
- DataLoader adaptive tuning (6h)
- Backend strategy (4h)
- Callback documentation (2h)
- Full end-to-end testing (5h)
- **Subtotal: 20 hours (2.5 FTE-days)**

**Total: ~42 hours (5-6 developer-weeks)**

---

## Success Criteria

- ✅ All 10 implementation tasks completed
- ✅ No regression in functionality
- ✅ Performance benchmarks show +0-15% improvement
- ✅ Code quality metrics improve to 8.5/10+ (all 3 phases)
- ✅ New test coverage >85%
- ✅ Zero high-severity technical debt remaining
- ✅ Production deployment confidence: HIGH

---

## Decision Points

1. **Backend Abstraction:** Remove (simplify) or implement JAX PoC?
   - **Recommendation:** Remove for now. Can add later if multi-backend is demanded.

2. **NVMe Prefetch Threading:** Sync or async (threading vs asyncio)?
   - **Recommendation:** Threading (simpler, doesn't block event loop).

3. **Profiling Overhead:** Always on or opt-in?
   - **Recommendation:** Opt-in flag (`--enable_profiling` during debug/benchmarking).

---

## Next Actions

1. [ ] Review and approve this roadmap
2. [ ] Assign developers to tasks
3. [ ] Create feature branches per task
4. [ ] Weekly sync to track progress
5. [ ] Run full validation suite after each task
6. [ ] Update documentation as we go

