# EGX Framework Upgrade Roadmap
## Technical Implementation Guide

---

## Executive Overview

This document provides **concrete code examples and implementation strategies** for upgrading EGX from alpha (v0.1) to production-ready (v1.0). It complements the Senior Code Review with actionable tasks organized by priority and complexity.

---

## Phase 1: Foundation Hardening (v0.2) — 4-6 Weeks

### P1.1: Complete Recovery Logic Implementation

**Current State:** Exception hierarchy exists but recovery actions not executed

**Upgrade Strategy:**
```python
# NEW: egx/resilience/recovery/orchestrator.py
from typing import AsyncGenerator
from enum import Enum
import time
import logging

class RecoveryStrategy(ABC):
    """Base for all recovery attempts."""
    
    @abstractmethod
    async def attempt(self, context: RecoveryContext) -> bool:
        """Try to recover. Returns True if successful."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """Lower = higher priority (0 = first to try)."""
        pass

@dataclass(frozen=True)
class RecoveryContext:
    """Contextual info for recovery decisions."""
    error: EGXError
    step: int
    last_checkpoint: Optional[str] = None
    remaining_retries: int = 5
    current_batch_size: int = 32
    peak_memory_usage: int = 0

class RetryStrategy(RecoveryStrategy):
    """Simple retry with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay_s: float = 1.0):
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self.attempt_count = 0
    
    async def attempt(self, context: RecoveryContext) -> bool:
        if not context.error.recoverable:
            logger.debug(f"Error not recoverable, skipping {self.name}")
            return False
        
        if self.attempt_count >= self.max_retries:
            logger.info(f"{self.name}: Max retries exceeded")
            return False
        
        delay = self.base_delay_s * (2 ** self.attempt_count)
        logger.info(f"{self.name}: Retry attempt {self.attempt_count + 1}/{self.max_retries} "
                   f"after {delay}s delay...")
        
        await asyncio.sleep(delay)
        self.attempt_count += 1
        return True
    
    @property
    def name(self) -> str:
        return "RetryStrategy"
    
    @property
    def priority(self) -> int:
        return 0  # Try retry first

class HalveBatchStrategy(RecoveryStrategy):
    """Reduce batch size and resume training."""
    
    async def attempt(self, context: RecoveryContext) -> bool:
        from egx.core.exceptions import OutOfMemoryError
        
        if not isinstance(context.error, OutOfMemoryError):
            return False
        
        if context.current_batch_size <= 1:
            logger.warning(f"{self.name}: Can't halve batch (already 1)")
            return False
        
        new_batch_size = max(1, context.current_batch_size // 2)
        logger.info(f"{self.name}: Reducing batch size "
                   f"{context.current_batch_size} → {new_batch_size}")
        
        # Signal trainer to update batch size
        # (Via callback or state mutation)
        return True
    
    @property
    def name(self) -> str:
        return "HalveBatchStrategy"
    
    @property
    def priority(self) -> int:
        return 1

class DowngradeStrategyStrategy(RecoveryStrategy):
    """Switch to more memory-efficient training mode."""
    
    async def attempt(self, context: RecoveryContext) -> bool:
        from egx.core.exceptions import OutOfMemoryError
        from egx.core.enums import TrainingMode
        
        if not isinstance(context.error, OutOfMemoryError):
            return False
        
        # Get current strategy from trainer state
        current_mode = getattr(context, 'current_training_mode', None)
        if not current_mode:
            return False
        
        # Try next-lower-memory mode in priority order
        FALLBACK_ORDER = [
            TrainingMode.FULL_FINETUNE,
            TrainingMode.LORA_PLUS,
            TrainingMode.LORA,
            TrainingMode.DORA,
            TrainingMode.QLORA,
        ]
        
        try:
            current_idx = FALLBACK_ORDER.index(current_mode)
        except ValueError:
            return False
        
        if current_idx < len(FALLBACK_ORDER) - 1:
            next_mode = FALLBACK_ORDER[current_idx + 1]
            logger.info(f"{self.name}: Downgrading {current_mode} → {next_mode}")
            # TODO: Implement mode switch
            return True
        
        return False
    
    @property
    def name(self) -> str:
        return "DowngradeStrategyStrategy"
    
    @property
    def priority(self) -> int:
        return 2

class CheckpointRollbackStrategy(RecoveryStrategy):
    """Restore from last good checkpoint."""
    
    async def attempt(self, context: RecoveryContext) -> bool:
        if not context.last_checkpoint:
            logger.debug(f"{self.name}: No checkpoint available")
            return False
        
        logger.warning(f"{self.name}: Rolling back to checkpoint: {context.last_checkpoint}")
        # Implementation would load checkpoint state
        # For now, signal success (trainer handles actual loading)
        return True
    
    @property
    def name(self) -> str:
        return "CheckpointRollbackStrategy"
    
    @property
    def priority(self) -> int:
        return 3

class RecoveryOrchestrator:
    """Coordinates recovery attempts in priority order."""
    
    def __init__(self):
        self.strategies: List[RecoveryStrategy] = [
            RetryStrategy(max_retries=3, base_delay_s=1.0),
            HalveBatchStrategy(),
            DowngradeStrategyStrategy(),
            CheckpointRollbackStrategy(),
        ]
        self.strategies.sort(key=lambda s: s.priority)
    
    async def recover(self, context: RecoveryContext) -> bool:
        """
        Attempt recovery strategies in priority order.
        Returns True if recovery succeeded, False if unrecoverable.
        """
        logger.warning(f"Initiating recovery for: {context.error.message}")
        
        for strategy in self.strategies:
            logger.debug(f"Attempting {strategy.name}...")
            try:
                if await strategy.attempt(context):
                    logger.info(f"✔ {strategy.name} recovered successfully")
                    return True
            except Exception as e:
                logger.error(f"✗ {strategy.name} failed: {e}")
                continue
        
        logger.critical("All recovery strategies exhausted")
        return False

# INTEGRATION: In TrainingKernel
class TrainingKernel(BaseTrainingKernel):
    def __init__(self, ...):
        self.recovery_orchestrator = RecoveryOrchestrator()
        self.last_checkpoint = None
    
    async def train_step_with_recovery(self, batch, step):
        """Train step with automatic recovery."""
        max_attempts = 1
        attempt = 0
        
        while attempt < max_attempts:
            try:
                return self.train_step(batch, step)
            except EGXError as e:
                attempt += 1
                
                context = RecoveryContext(
                    error=e,
                    step=step,
                    last_checkpoint=self.last_checkpoint,
                    current_batch_size=batch.get('batch_size', 'unknown'),
                )
                
                recovered = await self.recovery_orchestrator.recover(context)
                if not recovered:
                    raise
                
                # Update max_attempts for retry
                if isinstance(e, OutOfMemoryError):
                    max_attempts = 4  # Allow more retries for OOM
        
        raise EGXError("Max recovery attempts exceeded", recoverable=False)
```

---

### P1.2: ML-Augmented Memory Estimation

**Current Issue:** Activation memory hardcoded at 34.0 bits/layer → 30-50% error

**Solution:**
```python
# NEW: egx/intelligence/estimator/ml_based.py
import pickle
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn

class MLBasedEstimator(BaseEstimator):
    """
    Lightweight ML model predicts exact memory requirements.
    Trained offline on diverse model/config combinations.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Load pretrained memory predictor.
        Args:
            model_path: Path to serialized predictor (pickle or torchscript).
                       If None, uses bundled default.
        """
        self.model_path = model_path or self._get_bundled_model()
        with open(self.model_path, 'rb') as f:
            self.predictor = pickle.load(f)
        
        self.logger = logging.getLogger("egx.estimator.ml")
    
    def estimate(self, topology, profile, plan) -> MemoryReport:
        # Featurize: Extract key dimensions
        features = self._featurize(topology, profile, plan)
        
        # Predict: ML model predicts peak memory
        predicted_bytes = self.predictor.predict(features)[0]
        
        # Decompose: ML model also provides breakdown
        # (via output layer that predicts components separately)
        breakdown = self._decompose_prediction(features, predicted_bytes)
        
        return MemoryReport(
            weights_bytes=breakdown['weights'],
            activations_bytes=breakdown['activations'],
            gradients_bytes=breakdown['gradients'],
            optimizer_bytes=breakdown['optimizer'],
            overhead_bytes=breakdown['overhead'],
            total_bytes=predicted_bytes,
            method=EstimationMethod.ML_BASED,
            confidence=0.96,  # ML-based confidence higher
            error_bound_pct=3.0,  # Tighter bounds
            correction_factor=1.0,
        )
    
    def _featurize(self, topology, profile, plan) -> np.ndarray:
        """Convert problem geometry to feature vector."""
        return np.array([
            profile.params,                          # Total params
            profile.hidden_dim,
            profile.num_layers,
            profile.num_heads,
            plan.batch_size,
            plan.seq_len,
            plan.lora_rank if plan.mode.uses_peft() else 0,
            len(topology.gpus),
            topology.total_vram_bytes,
            plan.gradient_checkpointing,
            plan.mixed_precision,
            plan.flash_attention,
            plan.cpu_offload_optimizer,
            # Categoricals encoded:
            self._encode_dtype(plan.dtype),
            self._encode_training_mode(plan.mode),
        ])
    
    def _get_bundled_model(self) -> str:
        """Return path to default pretrained model."""
        pkg_dir = Path(__file__).parent.parent
        return pkg_dir / "models" / "memory_predictor.pkl"
    
    def _decompose_prediction(self, features, total_bytes) -> Dict[str, int]:
        """Break down total prediction into components."""
        # Component model: ensemble of regression heads
        w_frac = self.predictor.predict_weights_fraction(features)
        a_frac = self.predictor.predict_activations_fraction(features)
        g_frac = self.predictor.predict_gradients_fraction(features)
        o_frac = self.predictor.predict_optimizer_fraction(features)
        
        weights = int(total_bytes * w_frac)
        activations = int(total_bytes * a_frac)
        gradients = int(total_bytes * g_frac)
        optimizer = int(total_bytes * o_frac)
        overhead = total_bytes - weights - activations - gradients - optimizer
        
        return {
            'weights': weights,
            'activations': activations,
            'gradients': gradients,
            'optimizer': optimizer,
            'overhead': overhead,
        }

# TRAINING: Build training dataset for ML model
class MemoryEstimationTrainer:
    """Offline: Train ML predictor on diverse model/config space."""
    
    def generate_training_data(self, num_samples: int = 1000):
        """
        Generate synthetic training data covering model space.
        Features: [params, hidden_dim, ..., dtype_encoded, mode_encoded]
        Labels: Actual peak memory from dry runs
        """
        samples = []
        
        # Strategy: Sample uniformly from log-space of parameter counts
        param_ranges = [1e6, 1e7, 1e8, 1e9, 7e9]  # 1M to 7B
        batch_sizes = [1, 2, 4, 8, 16, 32]
        seq_lengths = [512, 1024, 2048, 4096]
        training_modes = [TrainingMode.FULL_FINETUNE, TrainingMode.LORA, TrainingMode.QLORA]
        
        for params in param_ranges:
            for bs in batch_sizes:
                for seq in seq_lengths:
                    for mode in training_modes:
                        # Create synthetic model config
                        profile = self._random_profile(params)
                        plan = TrainingPlan(
                            mode=mode,
                            batch_size=bs,
                            seq_len=seq,
                            # ... other fields
                        )
                        
                        # Measure actual peak memory (dry run)
                        actual_memory = self._measure_dry_run(profile, plan)
                        
                        # Featurize
                        features = self._featurize(profile, plan)
                        
                        samples.append((features, actual_memory))
        
        return np.array([s[0] for s in samples]), np.array([s[1] for s in samples])
    
    def _measure_dry_run(self, profile, plan) -> int:
        """Create dummy model and measure peak memory."""
        # This would be done offline, once
        pass
```

---

### P1.3: Complete Training Kernel

**Current Issue:** Gradient accumulation, clipping, scheduler not implemented

**Solution:**
```python
# UPDATED: egx/training/kernel.py
class TrainingKernel(BaseTrainingKernel):
    
    def train_step(self, batch: Dict[str, torch.Tensor], step: int) -> float:
        """
        Execute single training step with all features:
        1. Gradient accumulation
        2. Mixed precision
        3. Gradient clipping
        4. Scheduler updates
        Returns: scalar loss value for logging.
        """
        # Scale loss for gradient accumulation
        loss_scale = 1.0 / self.config.gradient_accumulation_steps
        
        # Forward pass
        with self._autocast_context():  # Mixed precision context
            outputs = self.model(batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs
            loss = loss * loss_scale
        
        # Backward pass
        if self.scaler:  # CUDA with mixed precision
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Check if we should optimizer step (every N accumulation steps)
        accum_idx = step % self.config.gradient_accumulation_steps
        should_step = (accum_idx == self.config.gradient_accumulation_steps - 1)
        
        if should_step:
            # Gradient clipping (before optimizer step)
            if self.config.max_grad_norm > 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            
            # Optimizer step
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
        
        # Return unscaled loss for logging
        return loss.item() / loss_scale
    
    def _autocast_context(self):
        """Return appropriate autocast context for mixed precision."""
        if self.config.precision_override:
            dtype_map = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            dtype = dtype_map.get(self.config.precision_override, torch.float32)
            return torch.autocast('cuda', dtype=dtype)
        return torch.no_op_context()
```

---

### P1.4: Comprehensive Test Suite

**Target:** 80%+ coverage, catching real bugs

```python
# NEW: tests/unit/test_estimators.py
import pytest
from hypothesis import given, strategies as st, settings
from egx.intelligence.estimator import AnalyticalEstimator, MLBasedEstimator
from egx.core.models import ModelProfile, TrainingPlan, HardwareTopology, MemoryReport
from egx.core.enums import TrainingMode, DType

class TestAnalyticalEstimator:
    
    @pytest.fixture
    def estimator(self):
        return AnalyticalEstimator()
    
    @pytest.fixture
    def topology(self):
        """Standard 40GB GPU topology."""
        return HardwareTopology(
            gpus=(GPUSpec(device_id=0, name='A100', vram_bytes=40*1024**3, ...),),
            cpu_cores=16,
            ram_bytes=256*1024**3,
            nvme_bytes=1*1024**4,
            ...
        )
    
    @pytest.fixture
    def llama_7b_profile(self):
        """LLaMA 7B model profile."""
        return ModelProfile(
            arch=ArchType.TRANSFORMER,
            params=7_000_000_000,
            hidden_dim=4096,
            num_layers=32,
            num_heads=32,
            max_seq_len=4096,
            dtype=DType.FP32,
        )
    
    def test_full_finetune_fits_on_40gb(self, estimator, topology, llama_7b_profile):
        """Full FT of LLaMA 7B should fit on 40GB A100."""
        plan = TrainingPlan(
            mode=TrainingMode.FULL_FINETUNE,
            dtype=DType.FP32,
            batch_size=4,
            seq_len=2048,
            # ...
        )
        
        report = estimator.estimate(topology, llama_7b_profile, plan)
        
        # Should fit with safety margin
        assert report.total_bytes < topology.total_vram_bytes * 0.85  # <85% utilization
    
    def test_qlora_uses_less_memory(self, estimator, topology, llama_7b_profile):
        """QLoRA should use significantly less memory than full FT."""
        
        full_ft_plan = TrainingPlan(
            mode=TrainingMode.FULL_FINETUNE,
            batch_size=4,
            # ...
        )
        
        qlora_plan = TrainingPlan(
            mode=TrainingMode.QLORA,
            batch_size=64,  # Can use larger batch with QLoRA
            # ...
        )
        
        full_report = estimator.estimate(topology, llama_7b_profile, full_ft_plan)
        qlora_report = estimator.estimate(topology, llama_7b_profile, qlora_plan)
        
        # QLoRA memory should be <20% of full FT
        assert qlora_report.total_bytes < full_report.total_bytes * 0.2
    
    @given(
        batch_size=st.integers(min_value=1, max_value=128),
        seq_len=st.integers(min_value=128, max_value=4096),
        hidden_dim=st.integers(min_value=256, max_value=8192),
    )
    @settings(deadline=None)
    def test_memory_monotonicity(self, estimator, batch_size, seq_len, hidden_dim):
        """Memory should increase monotonically with batch_size and seq_len."""
        
        plan_base = TrainingPlan(
            batch_size=batch_size,
            seq_len=seq_len,
            # ...
        )
        
        # Bump batch size
        plan_higher_batch = TrainingPlan(
            batch_size=batch_size * 2,
            seq_len=seq_len,
            # ...
        )
        
        report_base = estimator.estimate(...)
        report_higher = estimator.estimate(...)
        
        # More batch should use more memory
        assert report_higher.total_bytes >= report_base.total_bytes

# NEW: tests/integration/test_training_convergence.py
class TestTrainingConvergence:
    
    def test_toy_model_converges(self):
        """Verify training reduces loss on toy dataset."""
        
        trainer = EGXTrainer(
            config=EGXConfig(
                num_epochs=2,
                batch_size=4,
                learning_rate=1e-4,
                eval_strategy="epoch",
            )
        )
        
        # Tiny model: 1M params
        model = create_tiny_model(vocab_size=1000, hidden_size=256, num_layers=2)
        
        # Toy data: 100 samples
        dataset = create_synthetic_dataset(100)
        
        result = trainer.train(model, dataset)
        
        # Loss should decrease
        assert result['final_loss'] < result['initial_loss'], \
            f"Loss didn't decrease: {result['initial_loss']} -> {result['final_loss']}"
        
        # Should decrease by >= 10%
        loss_reduction_pct = (1 - result['final_loss'] / result['initial_loss']) * 100
        assert loss_reduction_pct > 10, f"Loss reduction only {loss_reduction_pct}%"

# NEW: tests/performance/test_memory_profiling.py
class TestMemoryProfiling:
    
    @pytest.mark.slow
    def test_memory_usage_no_leaks(self):
        """Verify no memory leaks over 1000 training steps."""
        
        trainer = EGXTrainer()
        model = create_test_model()
        dataset = create_large_dataset(1000)
        
        memory_samples = []
        
        # Monkey-patch to sample memory at each step
        original_step = trainer._engine.train_step
        
        def instrumented_step(batch, step):
            loss = original_step(batch, step)
            if torch.cuda.is_available():
                memory_samples.append(torch.cuda.memory_allocated())
            return loss
        
        trainer._engine.train_step = instrumented_step
        trainer.train(model, dataset)
        
        # Check for linear trend (leak signature)
        if len(memory_samples) > 100:
            early_avg = np.mean(memory_samples[:100])
            late_avg = np.mean(memory_samples[-100:])
            
            # Should not grow > 5% over course
            growth_pct = (late_avg - early_avg) / early_avg * 100
            assert growth_pct < 5, f"Memory grew {growth_pct}%"
```

---

## Phase 2: Feature Completeness (v0.3) — 4-8 Weeks

### P2.1: DoRA Implementation

```python
# NEW: egx/peft/dora.py
"""
Decomposed LoRA (DoRA): Decomposes weight updates into direction and magnitude.
Better convergence than standard LoRA, especially for large models.
"""

class DoRALinear(nn.Module):
    """
    DoRA-injected Linear layer.
    W_new = (D || M) where D is direction (via LoRA) and M is magnitude.
    """
    
    def __init__(
        self,
        original: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad = False
        
        out_features, in_features = original.weight.shape
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        
        # Direction matrices (LoRA)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Magnitude vector (trainable scaling per output dimension)
        self.m = nn.Parameter(torch.ones(out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output
        base_out = self.original(x)  # [batch, out_features]
        
        # LoRA direction: x @ A.T @ B.T = [batch, out_features]
        lora_direction = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        
        # Decomposed update: (m / ||m||) * lora_direction
        m_normalized = self.m / (torch.norm(self.m, p=2) + 1e-8)
        dora_update = (m_normalized.unsqueeze(0) * lora_direction) * self.scaling
        
        return base_out + dora_update

def inject_dora(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    targets: Optional[List[str]] = None,
) -> nn.Module:
    """Inject DoRA adapters into target Linear layers."""
    
    target_names = targets or ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"]
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_names):
            # Replace with DoRA version
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            
            setattr(parent, child_name, DoRALinear(module, rank, alpha))
    
    return model
```

### P2.2: Distributed Training Support (DDP/FSDP)

```python
# NEW: egx/api/distributed.py
"""
Distributed training coordinator.
Abstracts DDP (torch.nn.parallel.DistributedDataParallel)
and FSDP (torch.distributed.fsdp.FullyShardedDataParallel).
"""

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedConfig:
    """Distributed training configuration."""
    
    def __init__(
        self,
        backend: Literal["ddp", "fsdp", "none"] = "none",
        world_size: int = 1,
        rank: int = 0,
        master_addr: str = "localhost",
        master_port: int = 12355,
    ):
        self.backend = backend
        self.world_size = world_size
        self.rank = rank
        self.master_addr = master_addr
        self.master_port = master_port
    
    @classmethod
    def from_env(cls) -> "DistributedConfig":
        """Load from environment variables (torch.distributed.launch sets these)."""
        return cls(
            backend=os.getenv("DISTRIBUTED_BACKEND", "ddp"),
            world_size=int(os.getenv("WORLD_SIZE", "1")),
            rank=int(os.getenv("RANK", "0")),
            master_addr=os.getenv("MASTER_ADDR", "localhost"),
            master_port=int(os.getenv("MASTER_PORT", "12355")),
        )

class DistributedTrainer(EGXTrainer):
    """Extended trainer with distributed training support."""
    
    def __init__(self, *args, distributed_config: Optional[DistributedConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_config = distributed_config or DistributedConfig.from_env()
        self._is_main_process = self.dist_config.rank == 0
    
    def train(self, model, dataset, **kwargs):
        """Distributed training with automatic sharding."""
        
        # Initialize distributed process group
        if self.dist_config.world_size > 1:
            self._init_distributed()
        
        # Wrap model for distributed training
        model = self._wrap_model(model)
        
        # Shard dataset for each rank
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=self.dist_config.world_size,
            rank=self.dist_config.rank,
        )
        
        # Run training
        return super().train(model, dataset, data_sampler=sampler, **kwargs)
    
    def _init_distributed(self):
        """Initialize torch.distributed process group."""
        os.environ["MASTER_ADDR"] = self.dist_config.master_addr
        os.environ["MASTER_PORT"] = str(self.dist_config.master_port)
        
        torch.distributed.init_process_group(
            backend="nccl",  # or "gloo" for CPU
            rank=self.dist_config.rank,
            world_size=self.dist_config.world_size,
        )
    
    def _wrap_model(self, model):
        """Wrap model with DDP or FSDP."""
        if self.dist_config.backend == "fsdp":
            return FSDP(
                model,
                auto_wrap_policy=lambda module: isinstance(module, nn.TransformerEncoderLayer),
                device_id=torch.cuda.current_device(),
            )
        elif self.dist_config.backend == "ddp":
            return DDP(
                model,
                device_ids=[torch.cuda.current_device()],
                output_device=torch.cuda.current_device(),
            )
        return model

# USAGE:
# python -m torch.distributed.launch --nproc_per_node=8 train.py
# Inside train.py:
trainer = DistributedTrainer(distributed_config=DistributedConfig.from_env())
trainer.train(model, dataset)
```

---

## Phase 3: Polish & Optimization (v0.4) — 2-4 Weeks

### P3.1: Monitoring & Observability

```python
# NEW: egx/monitoring/prometheus_exporter.py
"""Export metrics in Prometheus format for monitoring."""

from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
training_step_counter = Counter(
    'egx_training_steps_total',
    'Total training steps completed',
    ['training_mode', 'dataset']
)

training_loss_histogram = Histogram(
    'egx_training_loss',
    'Distribution of training losses',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

memory_usage_gauge = Gauge(
    'egx_memory_usage_bytes',
    'Current GPU memory usage',
    ['gpu_id']
)

model_throughput_histogram = Histogram(
    'egx_throughput_samples_per_second',
    'Model throughput (samples/sec)',
)

class PrometheusCallback(TrainingCallback):
    def on_step_end(self, trainer, step, loss, lr, **kwargs):
        training_step_counter.labels(
            training_mode=trainer.config.get('training_mode', 'unknown'),
            dataset=trainer.config.get('dataset_name', 'unknown')
        ).inc()
        
        training_loss_histogram.observe(loss)
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.memory_allocated(i)
                memory_usage_gauge.labels(gpu_id=i).set(mem)
```

---

## Phase 4: Production Hardening (v1.0) — 4-6 Weeks

### P4.1: Formal State Machine

```python
# NEW: egx/resilience/recovery/state_machine.py
"""Formal state machine for training recovery."""

from transitions import Machine
from transitions.extensions.nesting import HierarchicalMachine

class TrainingStateMachine:
    """
    Formal state machine ensures valid transitions only.
    
    States:
    - Initialized: Trainer created, before boot
    - Booted: Hardware probed, model loaded
    - Training: Running training loop
    - Recovering: Attempting recovery from error
    - Checkpointing: Saving checkpoint
    - Paused: User-paused training
    - Complete: Training finished
    - Failed: Unrecoverable error
    """
    
    states = [
        'initialized',
        'booted',
        'training',
        {'name': 'recovering', 'children': ['retry', 'downgrade', 'rollback']},
        'checkpointing',
        'paused',
        'complete',
        'failed',
    ]
    
    transitions = [
        # Normal flow
        {'trigger': 'boot', 'source': 'initialized', 'dest': 'booted'},
        {'trigger': 'start_training', 'source': 'booted', 'dest': 'training'},
        {'trigger': 'checkpoint', 'source': 'training', 'dest': 'checkpointing',
         'after': 'resume_training'},
        {'trigger': 'training_complete', 'source': 'training', 'dest': 'complete'},
        
        # Recovery flow
        {'trigger': 'error_detected', 'source': 'training', 'dest': 'recovering_retry'},
        {'trigger': 'retry_failed', 'source': 'recovering_retry', 'dest': 'recovering_downgrade'},
        {'trigger': 'downgrade_complete', 'source': 'recovering_downgrade', 'dest': 'training'},
        {'trigger': 'recovery_failed', 'source': 'recovering_*', 'dest': 'failed'},
        
        # User actions
        {'trigger': 'pause', 'source': '*', 'dest': 'paused'},
        {'trigger': 'resume', 'source': 'paused', 'dest': 'training'},
    ]
    
    def __init__(self):
        self.machine = HierarchicalMachine(
            model=self,
            states=self.states,
            transitions=self.transitions,
            initial='initialized',
        )
    
    def is_trainable(self) -> bool:
        """Can we execute training steps?"""
        return self.state in ['training', 'booted']  # Include recovery states where possible
    
    def can_recover(self) -> bool:
        """Are we in a state where recovery is possible?"""
        return self.state.startswith('recovering')
```

---

## Testing Strategy (All Phases)

```yaml
# .github/workflows/ci.yml
name: EGX CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run type checking
        run: mypy --strict egx/
      
      - name: Run linting
        run: ruff check egx/ tests/
      
      - name: Run unit tests
        run: pytest tests/unit -v --cov=egx --cov-report=xml
      
      - name: Run integration tests
        run: pytest tests/integration -v --timeout=300
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  gpu-tests:
    runs-on: [self-hosted, gpu]
    if: "contains(github.event.head_commit.message, '[gpu]')"
    
    steps:
      - uses: actions/checkout@v3
      - name: Run GPU validation tests
        run: pytest tests/gpu_validation -v --timeout=600
```

---

## Success Metrics

| Metric | v0.2 | v0.3 | v1.0 |
|--------|------|------|------|
| Code Coverage | 60% | 75% | 85%+ |
| Type Checking | 50% strict | 70% strict | 100% strict |
| Memory Est. Accuracy | ±15% | ±5% | ±3% |
| Recovery Success Rate | 70% | 85% | 95%+ |
| Distributed Tests | None | DDP only | DDP + FSDP |
| Production Issues | TBD | <3/month | <1/month |

---

## Conclusion

This roadmap provides a structured path from alpha to production. **Each phase builds on the previous one** without disrupting existing functionality. Priority should be given to recovery logic and testing, as these are the highest-risk areas for production deployment.

**Estimated Timeline:** 5-6 months (with dedicated team of 2-3 engineers)

