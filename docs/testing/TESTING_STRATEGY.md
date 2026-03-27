# EGX Testing Phase - Comprehensive Strategy

## Overview
This testing phase establishes comprehensive end-to-end testing for the EGX framework, covering unit tests, integration tests, monitoring/metrics, and performance validation.

## Test Suite Structure

### 1. Unit Tests (`tests/unit/`)
Fast, isolated tests for individual components.

#### Core Components
- **`api/`** - Configuration, validation, interfaces
- **`core/`** - Utilities, device management, constants
- **`data/`** - Data loading, collation, streaming
- **`models/`** - Model registry, auto-detection
- **`peft/`** - LoRA, QLoRA, DoRA implementations
- **`plugins/`** - Flash attention, gradient checkpointing
- **`monitoring/`** - Metrics and telemetry (newly expanded)

#### Test Coverage
- Configuration validation
- Device detection and management
- Model loading and registry lookups
- PEFT adapter functionality
- Data collation and preprocessing
- Plugin initialization

**Run**: `python run_tests.py --suite unit`

### 2. Integration Tests (`tests/integration/`)
Multi-component tests validating end-to-end workflows.

#### Test Scenarios

**`test_lifecycle.py`** - Complete training workflow
- Training initialization
- Multi-epoch training
- Checkpoint saving/loading
- Resume from checkpoint
- Evaluation during/after training
- Memory management
- Performance validation
- Logging and reporting

**`test_large_models.py`** - Large model handling (if present)
- Memory estimation
- Batch size optimization
- Recovery and fallback
- Distributed training setup

**`test_data_pipeline.py`** - Data loading and processing
- DataLoader creation
- Batch processing
- Prefetching
- Streaming data
- Collation

**`test_peft_integration.py`** - PEFT adapter integration
- LoRA training
- QLoRA training
- DoRA training
- Adapter merging
- Model export

**Run**: `python run_tests.py --suite integration`

### 3. Monitoring & Metrics Tests (`tests/unit/monitoring/`)
NEW: Comprehensive tests for monitoring system.

#### `test_metrics.py`
- Metrics collection and aggregation
- Memory tracking (GPU/CPU)
- Performance metrics (throughput, latency)
- Metrics validation
- Metrics export (JSON)

#### `test_telemetry.py`
- Event tracking (training, checkpoints, errors)
- Telemetry aggregation
- Health monitoring
- Performance tracking
- Data serialization and retention

**Run**: `python run_tests.py --suite monitoring`

### 4. GPU Validation Tests (`tests/gpu_validation/`)
Tests requiring NVIDIA GPU.

#### Test Scenarios
- CUDA availability and version
- Memory profiling
- Kernel launch validation
- Distributed training setup
- GPU optimization validation

**Note**: Requires GPU. Use `--gpu` flag.

**Run**: `python run_tests.py --gpu`

### 5. Performance Benchmarking
Track performance metrics across versions.

- Throughput (samples/sec)
- Latency (ms/batch)
- Memory usage (MB)
- Training convergence speed
- Compilation time (for hardware-specific code)

## Running Tests

### Individual Test Suite
```bash
python run_tests.py --suite unit              # Unit tests only
python run_tests.py --suite integration       # Integration tests
python run_tests.py --suite monitoring        # Monitoring tests
python run_tests.py --suite large-models      # Large models
```

### Comprehensive Test Plan
Runs unit → integration → large-models → monitoring:
```bash
python run_tests.py --plan
```

### Coverage Report
Generate HTML coverage report:
```bash
python run_tests.py --report coverage
# HTML report generated in: htmlcov/index.html
```

### Verbose Output
```bash
python run_tests.py --suite unit -v
```

## Test Execution Strategy

### Phase 1: Rapid Feedback (< 5 min)
- Unit tests only
- Fast iteration during development
- **Command**: `python run_tests.py --suite unit`

### Phase 2: Integration Validation (< 15 min)
- Unit + integration tests
- Before committing changes
- **Command**: `python run_tests.py --suite unit integration monitoring`

### Phase 3: Pre-Release (< 30 min)
- All tests except GPU
- Coverage report generation
- **Command**: `python run_tests.py --plan`

### Phase 4: Full Validation (GPU machines only)
- All tests including GPU validation
- Performance benchmarking
- **Command**: `python run_tests.py --suite full --gpu`

## Monitoring Tests Details

### Metrics Collection (`test_metrics.py`)
Tests the monitoring framework's ability to:
- Initialize metrics collectors
- Update and aggregate metrics
- Track memory (GPU/CPU)
- Detect memory spikes
- Calculate throughput and latency
- Export to JSON
- Validate metric ranges (loss ≥ 0, accuracy ∈ [0,1])

### Telemetry Recording (`test_telemetry.py`)
Tests event tracking for:
- Training lifecycle (start, epoch, checkpoint, end)
- Error/warning recording
- Memory spike detection
- Performance trending
- System health monitoring
- Data aggregation and summarization
- Long-term retention policies

### Key Metrics Tracked
- **Training**: Loss, accuracy, learning rate
- **Performance**: Throughput (samples/sec), latency (ms/batch)
- **Memory**: GPU/CPU allocation, peak usage, spikes
- **System**: Temperature, power draw, disk usage
- **Convergence**: Loss trend, epoch-to-epoch improvement

## Integration Tests Details

### Lifecycle Tests (`test_lifecycle.py`)
Comprehensive training workflow covering:
1. Trainer initialization with config
2. Multi-epoch training loop
3. Validation during training
4. Checkpoint management:
   - Save/load functionality
   - Best model tracking
   - Metadata preservation
5. Resume from checkpoint
6. Post-training evaluation
7. Memory cleanup verification
8. Reproducibility with seeds

### Memory Management
- Pre/post training memory comparison
- Gradient accumulation memory profile
- Mixed precision (FP32 vs FP16) savings
- Batch size optimization

### Performance Validation
- Throughput maintenance
- Convergence tracking
- Accuracy improvement over epochs
- Learning rate schedule adherence

## Coverage Goals

### Target Coverage
| Component | Target |
|-----------|--------|
| Unit Tests | 85%+ |
| Integration Tests | 70%+ |
| Monitoring | 80%+ |
| Core APIs | 90%+ |
| Overall | 75%+ |

### Coverage Report
Generated by pytest-cov:
```
tests/unit/
├── monitoring/metrics.py: 85%
├── monitoring/telemetry.py: 82%
├── api/config.py: 92%
├── core/device.py: 88%
└── ...

Overall: 78%
```

View HTML report: `htmlcov/index.html`

## CI/CD Integration

### Pre-commit Testing
```bash
# Fast tests before commit
python run_tests.py --suite unit      # ~2-3 min
```

### Pull Request Testing
```bash
# Comprehensive but not GPU
python run_tests.py --plan            # ~20 min
python run_tests.py --report coverage # Generate report
```

### Release Testing
```bash
# Full test suite on GPU machines
python run_tests.py --suite full --gpu # ~40 min
```

## Performance Benchmarking

### Baseline Metrics (Expected)
- **Throughput**: 800+ samples/sec (on T4)
- **Latency (p50)**: ~2-5 ms/batch
- **Latency (p99)**: ~10-20 ms/batch
- **Memory**: 2GB baseline, +512MB per batch
- **Convergence**: Loss decreasing ~5-10% per epoch

### Regression Detection
Tests validate that:
- Throughput doesn't degrade > 5% from baseline
- Memory usage doesn't increase unexpectedly
- Errors don't appear
- Convergence rate stays consistent

## Troubleshooting

### Test Failures

#### OOM (Out of Memory)
- Reduce `batch_size` in test config
- Disable GPU tests if on low-memory system
- Check existing processes with `nvidia-smi`

#### CUDA Errors
- Verify PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU driver version: `nvidia-smi`
- Update CUDA drivers if needed

#### Timeout Issues
- Increase timeout in `run_tests.py`
- Run on faster hardware
- Check for system resource contention

### Performance Degradation
- Profile with `cProfile` or `torch.profiler`
- Check memory leaks with `memory_profiler`
- Compare metrics across versions

## Test Maintenance

### Adding New Tests
1. Create test file in appropriate directory
2. Follow naming convention: `test_*.py`
3. Use pytest or unittest
4. Add to test suite definition in `run_tests.py`
5. Document test purpose in docstring

### Updating Baselines
- After optimization improvements
- When hardware changes
- Quarterly baseline refresh

### Deprecating Tests
- Mark with `@pytest.mark.skip` with reason
- Document why deprecated
- Provide migration path to new tests

## Best Practices

### Write Tests That
✅ Test one thing clearly
✅ Have descriptive names
✅ Include docstrings/comments
✅ Are deterministic (same seed for randomness)
✅ Clean up after themselves
✅ Run fast (< 1s per test ideally)
✅ Are independent (no test order dependencies)

### Avoid Tests That
❌ Depend on external services unless mocked
❌ Have hardcoded paths
❌ Use system time without mocking
❌ Are flaky or non-deterministic
❌ Take > 10s unless integration test
❌ Print excessive output

## Next Steps

1. **Run baseline tests**: `python run_tests.py --plan`
2. **Generate coverage**: `python run_tests.py --report coverage`
3. **Review results**: Check for any failures or low coverage areas
4. **Iterate**: Fix failures and improve coverage
5. **Integrate into CI/CD**: Connect to GitHub Actions or similar

## Related Documentation

- [Architecture Decisions](../architecture/)
- [Training Guide](../training/)
- [Performance Tuning](../performance/)
- [Debugging Guide](../debugging/)

## Support

For test issues or questions:
1. Check this document first
2. Review test output logs
3. Run with `-v` flag for verbose output
4. File GitHub issue with minimal reproduction
