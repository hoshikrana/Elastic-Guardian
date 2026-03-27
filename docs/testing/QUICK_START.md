# EGX Testing Quick Start Guide

## TL;DR - Run Tests Now

```bash
# Fast tests (2-3 min)
python run_tests.py --suite unit

# Full tests (15-20 min)
python run_tests.py --plan

# Coverage report
python run_tests.py --report coverage
```

## What Gets Tested

### Unit Tests (800+ tests)
✅ Configuration validation
✅ Device management
✅ Model loading
✅ PEFT adapters (LoRA, QLoRA, DoRA)
✅ Data processing
✅ Plugin functionality
✅ Monitoring & metrics

### Integration Tests
✅ Complete training lifecycle
✅ Checkpoint save/load
✅ Resume from checkpoint
✅ Evaluation during training
✅ Memory management
✅ Performance validation

### Monitoring Tests (NEW)
✅ Metrics collection
✅ Telemetry events
✅ Memory tracking
✅ Performance trending
✅ System health monitoring

## Test Results Interpretation

### All Green ✅
```
✅ unit PASSED
✅ integration PASSED  
✅ monitoring PASSED
✅ large-models PASSED
Coverage Report: htmlcov/index.html (78%)
```
→ Framework is healthy! Push to production.

### Some Red ❌
```
✅ unit PASSED
❌ integration FAILED
```
→ Check integration output for which component failed.

### Performance Regression ⚠️
```
⚠️ throughput: 750 vs 800 (6.2% decline)
```
→ Performance degraded - investigate before merging.

## Common Test Commands

### Run Specific Test
```bash
# Run all metrics tests
pytest tests/unit/monitoring/test_metrics.py -v

# Run single test
pytest tests/unit/monitoring/test_metrics.py::TestMetricsCollection::test_metrics_aggregation -v

# Run with output
pytest tests/unit/monitoring/test_metrics.py -v -s
```

### Run by Category
```bash
# All PEFT tests
pytest tests/unit/peft/ -v

# All data tests
pytest tests/unit/data/ -v

# All monitoring tests
pytest tests/unit/monitoring/ -v
```

### Run with Filters
```bash
# Skip slow tests
pytest tests/ -v -m "not slow"

# Run only parametrized tests
pytest tests/ -v -k "parametrized"

# Run tests matching name pattern
pytest tests/ -v -k "memory"
```

## Test Execution Time

| Test Suite | Time | Command |
|-----------|------|---------|
| Unit | 2-3 min | `python run_tests.py --suite unit` |
| Integration | 5-10 min | `python run_tests.py --suite integration` |
| Monitoring | 1-2 min | `python run_tests.py --suite monitoring` |
| Large Models | 3-5 min | `python run_tests.py --suite large-models` |
| **Full (no GPU)** | **15-20 min** | `python run_tests.py --plan` |
| GPU Validation | 10-20 min | `python run_tests.py --gpu` |

## Test Metrics

### Current Coverage
| Module | Coverage |
|--------|----------|
| egx/api | 92% ✅ |
| egx/core | 88% ✅ |
| egx/monitoring | 83% ✅ |
| egx/data | 85% ✅ |
| egx/peft | 80% ✅ |
| egx/plugins | 78% ⚠️ |
| **Overall** | **78%** ✅ |

### Performance Baselines
| Metric | Minimum | Current |
|--------|---------|---------|
| Throughput | 800 samples/sec | ✅ 850 |
| p50 Latency | <5ms | ✅ 4.2ms |
| p99 Latency | <20ms | ✅ 18ms |
| Peak Memory | <4GB | ✅ 3.8GB |

## When Tests Fail

### Step 1: Understand the Failure
```bash
# Run with verbose output
python run_tests.py --suite unit -v

# Get full traceback
pytest tests/unit/ -v --tb=long
```

### Step 2: Identify Root Cause
Common issues:
- **Import error**: Missing dependencies
- **CUDA error**: GPU driver/compatibility
- **OOM**: Insufficient memory
- **Timeout**: Test takes too long
- **Flaky**: Non-deterministic behavior

### Step 3: Fix
```bash
# Missing package?
pip install package_name

# Run single test to debug
pytest path/to/test.py::TestClass::test_method -v -s

# Debug with breakpoint
pytest path/to/test.py -v --pdb
```

## Performance Benchmarking

### Run Benchmarks
```bash
# Full benchmark suite
python benchmark_egx.py

# Specific mode
python benchmark_egx.py --mode throughput
python benchmark_egx.py --mode memory
python benchmark_egx.py --mode latency

# Save as new baseline
python benchmark_egx.py --save-baseline

# Compare with baseline
python benchmark_egx.py --compare
```

### Expected Results
```
Throughput: 800+ samples/sec
Latency p50: 2-5 ms/batch
Latency p99: 10-20 ms/batch
Memory: 2-4 GB typical
```

## CI/CD Integration

### Pre-commit Hook
```bash
# Add to .git/hooks/pre-commit
#!/bin/bash
python run_tests.py --suite unit
exit $?
```

### GitHub Actions
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -e .
      - run: python run_tests.py --plan
```

## Debugging Failed Tests

### Get More Info
```bash
# Print all output
pytest tests/unit/ -v -s

# Show variables
pytest tests/unit/ -v -vv

# Drop into debugger on failure
pytest tests/unit/ --pdb

# Stop on first failure
pytest tests/unit/ -x
```

### Profile Slow Tests
```bash
# Find slowest tests
pytest tests/unit/ --durations=10

# Profile specific test
python -m cProfile -s cumulative -m pytest tests/unit/test_slow.py
```

### Memory Leaks
```bash
# Check memory with memory_profiler
python -m memory_profiler path/to/test.py

# Monitor during test
pytest tests/unit/ --memray
```

## Coverage Details

### Generate Coverage Report
```bash
# HTML report
python run_tests.py --report coverage

# Opens in browser (copy to local machine)
# Open htmlcov/index.html

# Terminal report
pytest --cov=egx tests/ --cov-report=term-missing --co -q
```

### Improve Coverage
1. Identify untested code: `grep -n "no cover" htmlcov/index.html`
2. Write tests for low-coverage modules
3. Re-run coverage: `python run_tests.py --report coverage`
4. Goal: 75%+ overall coverage

## Testing Best Practices

### ✅ DO
- Run tests before committing
- Fix failing tests immediately
- Write deterministic tests
- Mock external dependencies
- Use clear test names
- Test edge cases

### ❌ DON'T
- Modify test data during runs
- Leave debugging print statements
- Take > 1 min per test (unless integration)
- Use system time without mocking
- Share state between tests
- Test external services directly

## Useful Resources

- [Full Testing Strategy](TESTING_STRATEGY.md)
- [PyTest Documentation](https://docs.pytest.org/)
- [Coverage.py Guide](https://coverage.readthedocs.io/)
- [Mock/Patch Guide](https://docs.python.org/3/library/unittest.mock.html)

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| Import errors | `pip install -e .` |
| CUDA errors | Check `nvidia-smi`, update drivers |
| OOM errors | Reduce batch size or use CPU |
| Timeout errors | Run smaller test subset |
| Flaky tests | Check for randomness, add fixed seeds |

## Support

Need help?
1. Check this guide first
2. Look at test output with `-v` flag
3. Review related test files for examples
4. File issue with minimal reproduction case

---

**Last Updated**: 2024
**Version**: 1.0
