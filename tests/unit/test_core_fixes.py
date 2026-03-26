"""
EGX Comprehensive Test Suite — Unit Tests for Core Components.

Tests for:
- Recovery orchestration
- Memory estimation accuracy
- Training kernel functionality
- Strategy selection
"""

import pytest
import asyncio
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock

from egx.resilience.recovery.orchestrator import (
    RecoveryOrchestrator,
    RecoveryContext,
    RetryStrategy,
    HalveBatchStrategy,
)
from egx.core.exceptions import EGXError, OutOfMemoryError
from egx.core.models import (
    GPUSpec,
    HardwareTopology,
    ModelProfile,
    TrainingPlan,
    MemoryReport,
)
from egx.core.enums import (
    TrainingMode,
    DType,
    OptimizerType,
    ParallelStrategy,
    DeviceType,
    ArchType,
    InterconnectType,
    HardwareTier,
)
from egx.intelligence.estimator.improved_analytical import ImprovedAnalyticalEstimator


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def gpu_spec_a100():
    """40GB NVIDIA A100 specification."""
    return GPUSpec(
        device_id=0,
        name="A100-40GB",
        vram_bytes=40 * 1024**3,
        compute_capability=(8, 0),
        memory_bandwidth_gbps=2039.0,
        fp16_tflops=19500.0,
        bf16_tflops=19500.0,
        supports_flash_attn2=True,
        supports_fp8=True,
        nvlink_peer_ids=(),
        vendor="nvidia",
    )


@pytest.fixture
def topology_single_gpu(gpu_spec_a100):
    """Single GPU topology."""
    return HardwareTopology(
        gpus=(gpu_spec_a100,),
        cpu_cores=32,
        ram_bytes=256 * 1024**3,
        nvme_bytes=1 * 1024**4,
        nvme_seq_read_gbps=5.0,
        nvme_seq_write_gbps=4.0,
        pcie_bandwidth_gbps=32.0,
        gpu_interconnect_gbps=0,
        interconnect=InterconnectType.PCIE,
        node_count=1,
    )


@pytest.fixture
def llama_7b_profile():
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


@pytest.fixture
def training_plan_lora():
    """LoRA training plan configuration."""
    return TrainingPlan(
        mode=TrainingMode.LORA,
        dtype=DType.FP32,
        parallel_strategy=ParallelStrategy.NONE,
        batch_size=4,
        grad_accum_steps=1,
        seq_len=2048,
        gradient_checkpointing=False,
        mixed_precision=False,
        flash_attention=False,
        cpu_offload_optimizer=False,
        lora_rank=16,
        optimizer=OptimizerType.ADAMW,
    )


@pytest.fixture
def recovery_context_oom():
    """Recovery context for OOM error."""
    return RecoveryContext(
        error=OutOfMemoryError(
            msg="Requires 50GB, have 40GB",
        ),
        step=100,
        last_checkpoint_path="/checkpoint/step_90.pt",
        remaining_retries=3,
        current_batch_size=32,
        current_training_mode="lora",
        peak_memory_usage_bytes=40 * 1024**3,
    )


# ============================================================================
# Recovery Orchestrator Tests
# ============================================================================

class TestRecoveryOrchestrator:
    """Tests for recovery orchestration."""

    def test_orchestrator_initialization(self):
        """Orchestrator should initialize with all strategies in priority order."""
        orchestrator = RecoveryOrchestrator()
        
        assert len(orchestrator.strategies) == 4
        assert orchestrator.strategies[0].name == "RetryStrategy"
        assert orchestrator.strategies[1].name == "HalveBatchStrategy"
        assert orchestrator.strategies[2].name == "DowngradeStrategyStrategy"
        assert orchestrator.strategies[3].name == "CheckpointRollbackStrategy"
    
    @pytest.mark.asyncio
    async def test_retry_strategy_successful_recovery(self, recovery_context_oom):
        """Retry strategy should succeed on first attempt."""
        strategy = RetryStrategy(max_retries=3, base_delay_s=0.01)
        
        # First attempt should succeed
        result = await strategy.attempt(recovery_context_oom)
        assert result is True
        assert strategy.attempt_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_strategy_max_retries_exceeded(self, recovery_context_oom):
        """Retry strategy should fail after max_retries."""
        strategy = RetryStrategy(max_retries=2, base_delay_s=0.01)
        
        # Reach max retries
        for i in range(2):
            result = await strategy.attempt(recovery_context_oom)
            assert result is True
        
        # Next attempt should fail
        result = await strategy.attempt(recovery_context_oom)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_halve_batch_strategy_non_oom_error(self):
        """HalveBatch strategy should skip non-OOM errors."""
        strategy = HalveBatchStrategy()
        
        context = RecoveryContext(
            error=EGXError("Some error", recoverable=True),
            step=10,
            current_batch_size=32,
        )
        
        result = await strategy.attempt(context)
        assert result is False  # Not OOM, should skip
    
    @pytest.mark.asyncio
    async def test_halve_batch_strategy_reduces_batch(self, recovery_context_oom):
        """HalveBatch strategy should reduce batch size."""
        strategy = HalveBatchStrategy()
        context = recovery_context_oom
        
        result = await strategy.attempt(context)
        assert result is True
        assert strategy.halves_performed == 1
    
    @pytest.mark.asyncio
    async def test_orchestrator_recovery_flow(self, recovery_context_oom):
        """Orchestrator should try strategies in priority order."""
        orchestrator = RecoveryOrchestrator()
        
        # Should succeed with first strategy (Retry)
        result = await orchestrator.recover(recovery_context_oom)
        assert result is True
    
    def test_orchestrator_reset(self):
        """Orchestrator should reset strategy state."""
        orchestrator = RecoveryOrchestrator()
        
        # Mark some strategies as used
        retry_strat = orchestrator.strategies[0]
        retry_strat.attempt_count = 5
        
        # Reset
        orchestrator.reset()
        assert retry_strat.attempt_count == 0


# ============================================================================
# Memory Estimation Tests
# ============================================================================

class TestImprovedAnalyticalEstimator:
    """Tests for improved memory estimation."""
    
    def test_estimator_initialization(self):
        """Estimator should initialize without errors."""
        estimator = ImprovedAnalyticalEstimator()
        assert estimator is not None
    
    def test_full_finetune_memory_estimate(
        self, topology_single_gpu, llama_7b_profile, training_plan_lora
    ):
        """Full FT should estimate reasonable memory for 7B model on 40GB GPU."""
        estimator = ImprovedAnalyticalEstimator()
        
        plan = TrainingPlan(
            mode=TrainingMode.FULL_FINETUNE,
            dtype=DType.FP32,
            parallel_strategy=ParallelStrategy.NONE,
            batch_size=4,
            grad_accum_steps=1,
            seq_len=512,
            gradient_checkpointing=True,
            mixed_precision=False,
            flash_attention=False,
            cpu_offload_optimizer=False,
            lora_rank=0,  # Not used in full FT
            optimizer=OptimizerType.ADAMW,
        )
        
        report = estimator.estimate(topology_single_gpu, llama_7b_profile, plan)
        
        # Verify all memory components are calculated
        assert report.total_bytes > 0
        assert report.weights_bytes > 0
        assert report.activations_bytes > 0
        assert report.optimizer_bytes > 0
        assert report.confidence >= 0.85
    
    def test_lora_uses_less_memory_than_full_ft(
        self, topology_single_gpu, llama_7b_profile
    ):
        """LoRA should use significantly less memory than full FT."""
        estimator = ImprovedAnalyticalEstimator()
        
        plan_full_ft = TrainingPlan(
            mode=TrainingMode.FULL_FINETUNE,
            dtype=DType.FP32,
            parallel_strategy=ParallelStrategy.NONE,
            batch_size=4,
            grad_accum_steps=1,
            seq_len=512,
            gradient_checkpointing=True,
            mixed_precision=False,
            flash_attention=False,
            cpu_offload_optimizer=False,
            lora_rank=0,
            optimizer=OptimizerType.ADAMW,
        )
        
        plan_lora = TrainingPlan(
            mode=TrainingMode.LORA,
            dtype=DType.FP32,
            parallel_strategy=ParallelStrategy.NONE,
            batch_size=8,
            grad_accum_steps=1,
            seq_len=512,
            gradient_checkpointing=True,
            mixed_precision=False,
            flash_attention=False,
            cpu_offload_optimizer=False,
            lora_rank=16,
            optimizer=OptimizerType.ADAMW,
        )
        
        report_ft = estimator.estimate(topology_single_gpu, llama_7b_profile, plan_full_ft)
        report_lora = estimator.estimate(topology_single_gpu, llama_7b_profile, plan_lora)
        
        # LoRA gradients should be much smaller than full FT gradients
        assert report_lora.gradients_bytes < report_ft.gradients_bytes
    
    def test_gradient_checkpointing_reduces_activations(
        self, topology_single_gpu, llama_7b_profile
    ):
        """Gradient checkpointing should significantly reduce activation memory."""
        estimator = ImprovedAnalyticalEstimator()
        
        plan_no_ckpt = TrainingPlan(
            mode=TrainingMode.LORA,
            dtype=DType.FP32,
            parallel_strategy=ParallelStrategy.NONE,
            batch_size=16,
            grad_accum_steps=1,
            seq_len=2048,
            gradient_checkpointing=False,
            mixed_precision=False,
            flash_attention=False,
            cpu_offload_optimizer=False,
            lora_rank=16,
            optimizer=OptimizerType.ADAMW,
        )
        
        plan_with_ckpt = TrainingPlan(
            mode=TrainingMode.LORA,
            dtype=DType.FP32,
            parallel_strategy=ParallelStrategy.NONE,
            batch_size=16,
            grad_accum_steps=1,
            seq_len=2048,
            gradient_checkpointing=True,  # Enable checkpointing
            mixed_precision=False,
            flash_attention=False,
            cpu_offload_optimizer=False,
            lora_rank=16,
            optimizer=OptimizerType.ADAMW,
        )
        
        report_no_ckpt = estimator.estimate(topology_single_gpu, llama_7b_profile, plan_no_ckpt)
        report_with_ckpt = estimator.estimate(topology_single_gpu, llama_7b_profile, plan_with_ckpt)
        
        # Checkpointing should reduce activations by ~80%
        assert report_with_ckpt.activations_bytes < report_no_ckpt.activations_bytes * 0.3
        # Total should also be less (activations are large component)
        assert report_with_ckpt.total_bytes < report_no_ckpt.total_bytes * 0.7
    
    def test_memory_estimate_structure(
        self, topology_single_gpu, llama_7b_profile, training_plan_lora
    ):
        """Memory estimate should have all required components."""
        estimator = ImprovedAnalyticalEstimator()
        report = estimator.estimate(topology_single_gpu, llama_7b_profile, training_plan_lora)
        
        # All components should be present
        assert report.weights_bytes > 0
        assert report.activations_bytes > 0
        assert report.gradients_bytes > 0
        assert report.optimizer_bytes > 0
        assert report.overhead_bytes > 0
        assert report.total_bytes > 0
        
        # Total should be sum of components
        assert (
            report.total_bytes
            == (
                report.weights_bytes
                + report.activations_bytes
                + report.gradients_bytes
                + report.optimizer_bytes
                + report.overhead_bytes
            )
        )
        
        # Confidence should be high
        assert report.confidence >= 0.85
        # Error bound reasonable
        assert report.error_bound_pct <= 15.0


# ============================================================================
# Training Kernel Tests
# ============================================================================

class TestTrainingKernelIntegration:
    """Tests for training kernel with recovery."""
    
    def test_tiny_model_gradient_flow(self):
        """Verify gradients flow through tiny model."""
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create dummy batch
        x = torch.randn(2, 10)
        y = torch.randint(0, 10, (2,))
        
        # Forward pass
        loss_fn = nn.CrossEntropyLoss()
        outputs = model(x)
        loss = loss_fn(outputs, y)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are non-zero
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), \
                f"Gradient is zero for {name}"
        
        # Step
        optimizer.step()
        optimizer.zero_grad()
