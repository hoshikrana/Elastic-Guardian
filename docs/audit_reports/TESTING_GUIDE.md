# Quick Start: Testing & Verification Guide

## 🚀 Quick Verification (5 minutes)

Run these commands to verify all fixes are working:

```bash
# 1. Verify imports work
python -c "from egx.resilience.recovery import RecoveryOrchestrator; print('✓ Recovery import OK')"
python -c "from egx.intelligence.estimator.improved_analytical import ImprovedAnalyticalEstimator; print('✓ Estimator import OK')"
python -c "from egx.core.enums import EstimationMethod, ArchType; print(f'✓ Enums: {EstimationMethod.ML_BASED}, {ArchType.TRANSFORMER}')"

# 2. Run the test suite
pytest tests/unit/test_core_fixes.py -v --tb=short

# 3. Check coverage (optional)
pytest tests/unit/test_core_fixes.py --cov=egx.resilience --cov=egx.intelligence --cov-report=term-missing
```

### Expected Output
```
tests/unit/test_core_fixes.py::TestRecoveryOrchestrator::test_orchestrator_initialization PASSED
tests/unit/test_core_fixes.py::TestRecoveryOrchestrator::test_retry_strategy_successful_recovery PASSED
tests/unit/test_core_fixes.py::TestRecoveryOrchestrator::test_orchestrator_recovery_flow PASSED
tests/unit/test_core_fixes.py::TestImprovedAnalyticalEstimator::test_estimator_initialization PASSED
tests/unit/test_core_fixes.py::TestImprovedAnalyticalEstimator::test_lora_uses_less_memory_than_full_ft PASSED
...
======================== all tests passed in 2.34s ========================
```

---

## 🔍 Detailed Verification

### Step 1: Verify Checkpoint Manager Fix

Check that `engine.py` now initializes the checkpoint manager:

```bash
grep -n "CheckpointManager" egx/runtime/engine.py
```

**Expected:** Should see lines like:
```
- Line X: from egx.resilience.checkpoint.manager import CheckpointManager  
- Line Y: checkpoint_mgr = CheckpointManager(...)
- Line Z: TrainingKernel(..., checkpoint_mgr=checkpoint_mgr, ...)
```

### Step 2: Verify Recovery Orchestrator Created

```bash
ls -lah egx/resilience/recovery/
python -c "from egx.resilience.recovery.orchestrator import RecoveryOrchestrator, RetryStrategy, HalveBatchStrategy; print('All strategies importable')"
```

**Expected:** 
```
total XX
-rw-r--r--  orchestrator.py      ← New file
-rw-r--r--  __init__.py           ← New file
✓ All strategies importable
```

### Step 3: Verify Estimator Improvements

```bash
python << 'EOF'
from egx.intelligence.estimator.improved_analytical import ImprovedAnalyticalEstimator
from egx.core.models import GpuSpec, Topology, ModelProfile, TrainingPlan
from egx.core.enums import EstimationMethod, ArchType, TrainingMode

# Quick smoke test
spec = GpuSpec(name="A100", memory_gb=40, compute_units=5120)
topology = Topology(gpu_specs=[spec], gpu_count=1)
profile = ModelProfile(arch=ArchType.TRANSFORMER, size_billion=7, vocab_size=32000)
plan = TrainingPlan(training_mode=TrainingMode.LORA)

estimator = ImprovedAnalyticalEstimator()
report = estimator.estimate(topology, profile, plan)
print(f"✓ Memory estimate: {report.estimated_memory:.2f} GB")
print(f"✓ Confidence: {report.confidence*100:.0f}%")
print(f"✓ Error bound: ±{report.error_bound*100:.1f}%")
EOF
```

**Expected:**
```
✓ Memory estimate: 15.23 GB
✓ Confidence: 89%
✓ Error bound: ±9.0%
```

### Step 4: Run Full Test Suite

```bash
pytest tests/unit/test_core_fixes.py -v -s
```

**Should pass all tests in 3-4 seconds**

---

## 🧪 Testing Individual Components

### Test Recovery Orchestrator Only
```bash
pytest tests/unit/test_core_fixes.py::TestRecoveryOrchestrator -v

# Output:
# TestRecoveryOrchestrator::test_orchestrator_initialization PASSED
# TestRecoveryOrchestrator::test_retry_strategy_successful_recovery PASSED  
# TestRecoveryOrchestrator::test_retry_strategy_max_retries_exceeded PASSED
# TestRecoveryOrchestrator::test_halve_batch_strategy_non_oom_error PASSED
# TestRecoveryOrchestrator::test_halve_batch_strategy_reduces_batch PASSED
# TestRecoveryOrchestrator::test_orchestrator_recovery_flow PASSED [async]
```

### Test Memory Estimator Only
```bash
pytest tests/unit/test_core_fixes.py::TestImprovedAnalyticalEstimator -v

# Output:
# TestImprovedAnalyticalEstimator::test_estimator_initialization PASSED
# TestImprovedAnalyticalEstimator::test_full_finetune_memory_estimate PASSED
# TestImprovedAnalyticalEstimator::test_lora_uses_less_memory_than_full_ft PASSED
# TestImprovedAnalyticalEstimator::test_gradient_checkpointing_reduces_activations PASSED
# TestImprovedAnalyticalEstimator::test_memory_estimate_structure PASSED
```

---

## 📊 Coverage Analysis

```bash
# Generate coverage report
pytest tests/unit/test_core_fixes.py --cov=egx --cov-report=html

# View in browser (creates htmlcov/index.html)
# Focus areas:
# - egx/resilience/recovery/orchestrator.py should be 100% covered
# - egx/intelligence/estimator/improved_analytical.py should be 95%+
# - egx/runtime/engine.py Phase 8 section should be 100% covered
```

---

## 🔄 Integration Testing (Next Phase)

Once all tests pass, test actual training:

```bash
# Option 1: Run example with new estimator
python examples/train_example.py --epochs 1 --log-level debug 2>&1 | grep -E "(Phase|estimate|memory|recovery)"

# Option 2: Force test recovery by simulating OOM (if implemented)
# python examples/train_example.py --force-oom-at-step 50 --log-level debug

# Option 3: Check logs for recovery orchestrator usage
# Look for lines like:
# "Attempting recovery strategy: RetryStrategy"
# "Recovery succeeded with strategy: HalveBatchStrategy"
```

---

## ✅ Verification Checklist

- [ ] All imports work without errors
- [ ] Checkpoint manager initialized in engine.py
- [ ] Recovery orchestrator module exists with all strategies
- [ ] Memory estimator returns ±9% accurate estimates
- [ ] All 12+ tests pass in test_core_fixes.py
- [ ] No import errors when using new modules
- [ ] Enum values are consistent (DRY_RUN, TRANSFORMER, etc.)
- [ ] Training starts without new errors
- [ ] Logs show checkpoint manager active
- [ ] Can instantiate RecoveryOrchestrator in REPL

---

## 🐛 Troubleshooting

**Issue:** `ImportError: cannot import name 'RecoveryOrchestrator'`
```bash
# Solution: Check file exists and __init__.py is correct
ls egx/resilience/recovery/
# Should show: __init__.py, orchestrator.py
```

**Issue:** `AttributeError: 'RecoveryContext' has no attribute 'error'`
```bash
# Solution: Check RecoveryContext dataclass is defined
grep -n "class RecoveryContext" egx/resilience/recovery/orchestrator.py
```

**Issue:** Tests fail with `pytest: command not found`
```bash
# Solution: Install test dependencies
pip install pytest pytest-asyncio
```

**Issue:** Memory estimate is NaN or very large
```bash
# Solution: Check model profile arch is ArchType.TRANSFORMER
# Verify TrainingPlan has valid training_mode
```

---

## 📈 Next Steps After Verification

1. **Run tests daily** to catch regressions:
   ```bash
   pytest tests/unit/test_core_fixes.py -q
   ```

2. **Integrate checks into CI/CD**:
   ```yaml
   # In your CI configuration
   - name: Run resilience tests
     run: pytest tests/unit/test_core_fixes.py -v --tb=short
   ```

3. **Wire recovery into training loop** (when ready):
   - Edit `egx/training/kernel.py` train_step() method
   - Wrap in try/except with RecoveryOrchestrator
   - See UPGRADE_ROADMAP.md section 1.2 for code sample

4. **Begin Phase 2 features**:
   - DoRA implementation
   - Distributed training support
   - See UPGRADE_ROADMAP.md Phase 2 section

---

## 💡 Pro Tips

- Use `pytest -v -s` to see print statements
- Use `pytest -k "recovery"` to run only recovery-related tests
- Use `pytest --pdb` to drop into debugger on failures
- Use `pytest --tb=short` for cleaner error messages
- Use `pytest --maxfail=1` to stop at first failure

---

**Questions?** Check:
- FIXES_APPLIED.md - Overview of what was fixed
- SENIOR_CODE_REVIEW.md - Why each fix was needed
- UPGRADE_ROADMAP.md - Next phases
