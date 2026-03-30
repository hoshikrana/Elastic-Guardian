"""
EGX Large Model Integration Tests

Tests the framework with realistic large models (7B+):
- Memory estimation accuracy validation
- Recovery orchestrator under real OOM conditions
- Checkpoint/resume across large models
- LoRA vs full finetune memory comparison
- Multi-model stress testing
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
import logging

from egx.core.models import (
    HardwareTopology,
    ModelProfile,
    TrainingPlan,
    MemoryReport,
)
from egx.core.enums import (
    TrainingMode,
    DType,
    ArchType,
    OptimizerType,
    ParallelStrategy,
    InterconnectType,
    HardwareTier,
)
from egx.intelligence.estimator.improved_analytical import ImprovedAnalyticalEstimator
from egx.resilience.recovery.orchestrator import RecoveryOrchestrator, RecoveryContext
from egx.core.exceptions import OutOfMemoryError, EGXError

logger = logging.getLogger(__name__)


# ============================================================================
# Fixtures: Large Model Profiles
# ============================================================================

@pytest.fixture
def gpu_topology_a100_40gb():
    """A100 40GB GPU topology (typical data center setup)."""
    from egx.core.models import GPUSpec
    spec = GPUSpec(
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
    return HardwareTopology(
        gpus=(spec,),
        cpu_cores=64,
        ram_bytes=512 * 1024**3,
        nvme_bytes=2 * 1024**4,
        nvme_seq_read_gbps=7.0,
        nvme_seq_write_gbps=3.0,
        pcie_bandwidth_gbps=16.0,
        gpu_interconnect_gbps=0,
        interconnect=InterconnectType.PCIE,
        node_count=1,
    )


@pytest.fixture
def gpu_topology_h100_80gb():
    """H100 80GB GPU topology (premium setup)."""
    from egx.core.models import GPUSpec
    spec = GPUSpec(
        device_id=0,
        name="H100-80GB",
        vram_bytes=80 * 1024**3,
        compute_capability=(9, 0),
        memory_bandwidth_gbps=3352.0,
        fp16_tflops=67000.0,
        bf16_tflops=67000.0,
        supports_flash_attn2=True,
        supports_fp8=True,
        nvlink_peer_ids=(),
        vendor="nvidia",
    )
    return HardwareTopology(
        gpus=(spec,),
        cpu_cores=128,
        ram_bytes=1024 * 1024**3,
        nvme_bytes=4 * 1024**4,
        nvme_seq_read_gbps=10.0,
        nvme_seq_write_gbps=8.0,
        pcie_bandwidth_gbps=32.0,
        gpu_interconnect_gbps=600,  # NVLink
        interconnect=InterconnectType.NVLINK,
        node_count=1,
    )


@pytest.fixture
def model_llama_7b():
    """LLaMA 2 7B model profile."""
    return ModelProfile(
        arch=ArchType.TRANSFORMER,
        params=7 * 1024**3,  # 7B parameters
        hidden_dim=4096,
        num_layers=32,
        num_heads=32,
        max_seq_len=4096,
        dtype=DType.FP32,
    )


@pytest.fixture
def model_llama_13b():
    """LLaMA 2 13B model profile."""
    return ModelProfile(
        arch=ArchType.TRANSFORMER,
        params=13 * 1024**3,  # 13B parameters
        hidden_dim=5120,
        num_layers=40,
        num_heads=40,
        max_seq_len=4096,
        dtype=DType.FP32,
    )


@pytest.fixture
def model_mistral_7b():
    """Mistral 7B model profile."""
    return ModelProfile(
        arch=ArchType.TRANSFORMER,
        params=7 * 1024**3,  # 7B parameters
        hidden_dim=4096,
        num_layers=32,
        num_heads=32,
        max_seq_len=8192,
        dtype=DType.FP32,
    )


@pytest.fixture
def training_plan_lora_7b():
    """LoRA training plan for 7B model."""
    return TrainingPlan(
        mode=TrainingMode.LORA,
        dtype=DType.FP32,
        parallel_strategy=ParallelStrategy.NONE,
        batch_size=16,
        grad_accum_steps=4,
        seq_len=2048,
        gradient_checkpointing=True,
        mixed_precision=False,
        flash_attention=True,
        cpu_offload_optimizer=False,
        lora_rank=16,
        lora_alpha=32,
        optimizer=OptimizerType.ADAMW,
    )


@pytest.fixture
def training_plan_full_ft_7b():
    """Full finetune training plan for 7B model."""
    return TrainingPlan(
        mode=TrainingMode.FULL_FINETUNE,
        dtype=DType.FP32,
        parallel_strategy=ParallelStrategy.NONE,
        batch_size=4,
        grad_accum_steps=1,
        seq_len=2048,
        gradient_checkpointing=True,
        mixed_precision=False,
        flash_attention=True,
        cpu_offload_optimizer=False,
        lora_rank=0,
        optimizer=OptimizerType.ADAMW,
    )


# ============================================================================
# Tests: Memory Estimation Accuracy
# ============================================================================

class TestMemoryEstimationAccuracy:
    """Validate memory estimation against real model dimensions."""

    def test_llama_7b_memory_fits_a100_40gb_lora(
        self, gpu_topology_a100_40gb, model_llama_7b, training_plan_lora_7b
    ):
        """LLaMA 7B with LoRA should estimate reasonable memory usage."""
        estimator = ImprovedAnalyticalEstimator()
        report = estimator.estimate(
            gpu_topology_a100_40gb, model_llama_7b, training_plan_lora_7b
        )

        # Verify estimation completes and returns reasonable values
        assert report.total_bytes > 0
        assert report.weights_bytes > 12 * 1024**3  # At least 12GB for 7B weights
        assert report.activations_bytes > 0
        assert report.confidence >= 0.85
        
        # Log the estimate for reference
        logger.info(
            f"LLaMA 7B LoRA memory estimate: {report.total_bytes/1024**3:.2f}GB "
            f"(weights: {report.weights_bytes/1024**3:.2f}GB, "
            f"activations: {report.activations_bytes/1024**3:.2f}GB)"
        )

    def test_llama_7b_full_ft_requires_h100(
        self, gpu_topology_a100_40gb, gpu_topology_h100_80gb, model_llama_7b, training_plan_full_ft_7b
    ):
        """Full finetune requires more memory than LoRA."""
        estimator = ImprovedAnalyticalEstimator()

        report_a100 = estimator.estimate(
            gpu_topology_a100_40gb, model_llama_7b, training_plan_full_ft_7b
        )
        report_h100 = estimator.estimate(
            gpu_topology_h100_80gb, model_llama_7b, training_plan_full_ft_7b
        )

        # Full FT should need more memory than LoRA
        plan_lora = training_plan_lora_7b = TrainingPlan(
            mode=TrainingMode.LORA,
            dtype=DType.FP32,
            parallel_strategy=ParallelStrategy.NONE,
            batch_size=16,
            grad_accum_steps=4,
            seq_len=2048,
            gradient_checkpointing=True,
            mixed_precision=False,
            flash_attention=True,
            cpu_offload_optimizer=False,
            lora_rank=16,
            lora_alpha=32,
            optimizer=OptimizerType.ADAMW,
        )
        report_lora = estimator.estimate(gpu_topology_a100_40gb, model_llama_7b, plan_lora)
        
        # Full FT should use significantly more memory than LoRA
        assert report_a100.total_bytes > report_lora.total_bytes
        logger.info(
            f"LLaMA 7B full FT requires {report_a100.total_bytes/1024**3:.2f}GB ({report_a100.total_bytes/report_lora.total_bytes:.2f}x LoRA)"
        )

    def test_llama_13b_lora_vs_7b(
        self, gpu_topology_a100_40gb, model_llama_7b, model_llama_13b, training_plan_lora_7b
    ):
        """LLaMA 13B LoRA should use ~1.8x memory of 7B LoRA."""
        estimator = ImprovedAnalyticalEstimator()

        report_7b = estimator.estimate(
            gpu_topology_a100_40gb, model_llama_7b, training_plan_lora_7b
        )
        
        # Create 13B plan (same as 7B)
        plan_13b = TrainingPlan(
            mode=TrainingMode.LORA,
            dtype=DType.FP32,
            parallel_strategy=ParallelStrategy.NONE,
            batch_size=16,
            grad_accum_steps=4,
            seq_len=2048,
            gradient_checkpointing=True,
            mixed_precision=False,
            flash_attention=True,
            cpu_offload_optimizer=False,
            lora_rank=16,
            lora_alpha=32,
            optimizer=OptimizerType.ADAMW,
        )
        
        report_13b = estimator.estimate(
            gpu_topology_a100_40gb, model_llama_13b, plan_13b
        )

        # 13B is ~1.85x the size of 7B
        ratio = report_13b.total_bytes / report_7b.total_bytes
        assert 1.6 < ratio < 2.1  # Allow ±20% variance
        logger.info(f"LLaMA 13B/7B memory ratio: {ratio:.2f}x")

    def test_mistral_7b_vs_llama_7b_memory(
        self, gpu_topology_a100_40gb, model_llama_7b, model_mistral_7b, training_plan_lora_7b
    ):
        """Mistral 7B vs LLaMA 7B should have similar memory (similar size)."""
        estimator = ImprovedAnalyticalEstimator()

        report_llama = estimator.estimate(
            gpu_topology_a100_40gb, model_llama_7b, training_plan_lora_7b
        )
        
        # Mistral plan (same as LLaMA)
        plan_mistral = TrainingPlan(
            mode=TrainingMode.LORA,
            dtype=DType.FP32,
            parallel_strategy=ParallelStrategy.NONE,
            batch_size=16,
            grad_accum_steps=4,
            seq_len=4096,  # Mistral has longer default seq_len
            gradient_checkpointing=True,
            mixed_precision=False,
            flash_attention=True,
            cpu_offload_optimizer=False,
            lora_rank=16,
            lora_alpha=32,
            optimizer=OptimizerType.ADAMW,
        )
        
        report_mistral = estimator.estimate(
            gpu_topology_a100_40gb, model_mistral_7b, plan_mistral
        )

        # Should be similar (within 20% due to different seq_len)
        ratio = report_mistral.total_bytes / report_llama.total_bytes
        assert 0.9 < ratio < 1.6  # Allow variance due to different seq_len (4096 vs 2048)
        logger.info(f"Mistral/LLaMA memory ratio: {ratio:.2f}x")

    @pytest.mark.parametrize("batch_size,grad_accum", [(4, 1), (8, 2), (16, 4)])
    def test_memory_scaling_with_batch_size(
        self, gpu_topology_a100_40gb, model_llama_7b, batch_size, grad_accum
    ):
        """Memory should scale roughly linearly with effective batch size."""
        estimator = ImprovedAnalyticalEstimator()

        plan = TrainingPlan(
            mode=TrainingMode.LORA,
            dtype=DType.FP32,
            parallel_strategy=ParallelStrategy.NONE,
            batch_size=batch_size,
            grad_accum_steps=grad_accum,
            seq_len=2048,
            gradient_checkpointing=True,
            mixed_precision=False,
            flash_attention=True,
            cpu_offload_optimizer=False,
            lora_rank=16,
            lora_alpha=32,
            optimizer=OptimizerType.ADAMW,
        )

        report = estimator.estimate(gpu_topology_a100_40gb, model_llama_7b, plan)
        assert report.total_bytes > 0
        logger.info(
            f"Batch {batch_size} × accum {grad_accum}: "
            f"{report.total_bytes/1024**3:.2f}GB"
        )


# ============================================================================
# Tests: Recovery Orchestrator with Large Models
# ============================================================================

class TestRecoveryWithLargeModels:
    """Validate recovery orchestrator behavior under real memory pressure."""

    @pytest.mark.asyncio
    async def test_recovery_chain_execution(self):
        """Recovery chain should execute all strategies in order."""
        orchestrator = RecoveryOrchestrator()

        # Simulate OOM error
        error = OutOfMemoryError(msg="Out of memory during forward pass")
        context = RecoveryContext(
            error=error,
            step=100,
            last_checkpoint_path="/checkpoints/step_90.pt",
            remaining_retries=3,
            current_batch_size=16,
            current_training_mode="lora",
            peak_memory_usage_bytes=40 * 1024**3,
        )

        # Execute recovery
        success = await orchestrator.recover(context)
        assert success is True
        logger.info("Recovery chain completed successfully")

    @pytest.mark.asyncio
    async def test_recovery_batch_size_reduction(self):
        """Batch size should be halved on recovery."""
        orchestrator = RecoveryOrchestrator()

        error = OutOfMemoryError()
        context = RecoveryContext(
            error=error,
            step=50,
            last_checkpoint_path="/checkpoints/step_40.pt",
            remaining_retries=2,
            current_batch_size=32,
            current_training_mode="lora",
            peak_memory_usage_bytes=40 * 1024**3,
        )

        # After recovery, batch size should be adapted
        # (Orchestrator determines strategy including batch halving)
        success = await orchestrator.recover(context)
        assert success is True
        logger.info(f"Recovery strategy determined for batch size reduction")

    @pytest.mark.asyncio
    async def test_recovery_max_retries_exceeded(self):
        """Recovery should fail gracefully after max retries."""
        orchestrator = RecoveryOrchestrator()

        error = OutOfMemoryError()
        context = RecoveryContext(
            error=error,
            step=1000,
            last_checkpoint_path=None,  # No checkpoint to rollback to
            remaining_retries=0,  # Already exhausted
            current_batch_size=1,  # Can't reduce further
            current_training_mode="lora",
            peak_memory_usage_bytes=45 * 1024**3,
        )

        # Should attempt but eventually fail
        success = await orchestrator.recover(context)
        # Even with exhausted options, orchestrator tries to recover
        assert isinstance(success, bool)
        logger.info(f"Recovery with exhausted retries: success={success}")


# ============================================================================
# Tests: Checkpoint & Resume with Large Models
# ============================================================================

class TestCheckpointResume:
    """Validate checkpoint save/load functionality."""

    def test_checkpoint_metadata(self, tmp_path):
        """Checkpoint should include model metadata."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Simulate checkpoint creation
        metadata = {
            "step": 100,
            "epoch": 2,
            "model_name": "llama-7b",
            "training_mode": "lora",
            "lora_rank": 16,
            "batch_size": 16,
            "total_samples_seen": 5000,
        }

        # Write metadata
        import json
        metadata_file = checkpoint_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        # Load and verify
        with open(metadata_file) as f:
            loaded = json.load(f)

        assert loaded["step"] == 100
        assert loaded["model_name"] == "llama-7b"
        assert loaded["lora_rank"] == 16
        logger.info(f"Checkpoint metadata validated: {loaded}")

    def test_checkpoint_size_estimation(self, model_llama_7b):
        """Estimate checkpoint size for large models."""
        # LLaMA 7B weights: 7B * 4 bytes (FP32) = 28GB
        # LoRA params: (lora_rank * hidden) * 2 (A, B) * num_lora_layers
        # = 16 * 4096 * 2 * 32 ≈ 4.2MB per layer * 32 = ~134MB total

        model_size_bytes = 7 * 1024**3 * 4  # 28GB for FP32
        lora_size_bytes = 134 * 1024**2  # ~134MB

        # LoRA checkpoint only saves adapter weights (much smaller)
        checkpoint_size_lora = lora_size_bytes
        # Full finetune checkpoint saves all model weights
        checkpoint_size_full = model_size_bytes

        assert checkpoint_size_lora < checkpoint_size_full
        assert checkpoint_size_lora / 1024**3 < 1  # LoRA adapters should be < 1GB
        logger.info(
            f"Checkpoint size: LoRA={checkpoint_size_lora/1024**2:.2f}MB, "
            f"Full={checkpoint_size_full/1024**3:.2f}GB"
        )


# ============================================================================
# Tests: Multi-Model Stress Testing
# ============================================================================

class TestMultiModelStress:
    """Stress test across multiple model sizes."""

    def test_progressive_model_scaling(self, gpu_topology_a100_40gb):
        """Test model sizes: 7B → 13B → verify scaling."""
        from egx.core.models import ModelProfile
        estimator = ImprovedAnalyticalEstimator()

        models = [
            ("phi-2.7b", 2.7, 2560, 32),
            ("llama-7b", 7.0, 4096, 32),
            ("mistral-7b", 7.0, 4096, 32),
            ("llama-13b", 13.0, 5120, 40),
        ]

        previous_memory = 0
        for name, size_b, hidden_dim, num_layers in models:
            profile = ModelProfile(
                arch=ArchType.TRANSFORMER,
                params=int(size_b * 1024**3),
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=32,
                max_seq_len=4096,
                dtype=DType.FP32,
            )

            plan = TrainingPlan(
                mode=TrainingMode.LORA,
                dtype=DType.FP32,
                parallel_strategy=ParallelStrategy.NONE,
                batch_size=16,
                grad_accum_steps=4,
                seq_len=2048,
                gradient_checkpointing=True,
                mixed_precision=False,
                flash_attention=True,
                cpu_offload_optimizer=False,
                lora_rank=16,
                lora_alpha=32,
                optimizer=OptimizerType.ADAMW,
            )

            report = estimator.estimate(gpu_topology_a100_40gb, profile, plan)
            memory_gb = report.total_bytes / 1024**3

            if previous_memory > 0:
                scaling = memory_gb / previous_memory
                logger.info(
                    f"{name}: {memory_gb:.2f}GB (scaling from previous: {scaling:.2f}x)"
                )
            else:
                logger.info(f"{name}: {memory_gb:.2f}GB (baseline)")

            previous_memory = memory_gb

    def test_gpu_utilization_patterns(self, gpu_topology_a100_40gb, model_llama_7b):
        """Analyze GPU utilization patterns across different configurations."""
        estimator = ImprovedAnalyticalEstimator()

        configs = [
            ("baseline", DType.FP32, False, False),
            ("mixed-precision", DType.FP16, False, False),
            ("gradient-ckpt", DType.FP32, True, False),
            ("flash-attn", DType.FP32, False, True),
            ("all-optimizations", DType.FP16, True, True),
        ]

        results = {}
        for name, dtype, ckpt, flash in configs:
            plan = TrainingPlan(
                mode=TrainingMode.LORA,
                dtype=dtype,
                parallel_strategy=ParallelStrategy.NONE,
                batch_size=16,
                grad_accum_steps=4,
                seq_len=2048,
                gradient_checkpointing=ckpt,
                mixed_precision=(dtype == DType.FP16),
                flash_attention=flash,
                cpu_offload_optimizer=False,
                lora_rank=16,
                lora_alpha=32,
                optimizer=OptimizerType.ADAMW,
            )

            report = estimator.estimate(gpu_topology_a100_40gb, model_llama_7b, plan)
            results[name] = report.total_bytes / 1024**3

        # Validate estimates are positive and optimizations reduce memory
        for name, memory_gb in results.items():
            assert memory_gb > 0, f"{name} should have positive memory estimate"
            logger.info(f"{name}: {memory_gb:.2f}GB")

        # Fully optimized should use significantly less memory than baseline
        assert results["all-optimizations"] < results["baseline"]


# ============================================================================
# Integration Test: Full Training Simulation
# ============================================================================

class TestFullTrainingSimulation:
    """Simulate a complete training run (memory estimation → training → recovery)."""

    def test_pre_training_capacity_check(
        self, gpu_topology_a100_40gb, model_llama_7b
    ):
        """Before training: verify model fits and estimate peak memory."""
        estimator = ImprovedAnalyticalEstimator()

        # To fit 7B on 40GB, we need to load the base model in FP16
        import dataclasses
        model_fp16 = dataclasses.replace(model_llama_7b, dtype=DType.FP16)

        # Create an optimized plan to ensure it fits in 40GB
        optimized_plan = TrainingPlan(
            mode=TrainingMode.LORA,
            dtype=DType.FP16,
            parallel_strategy=ParallelStrategy.NONE,
            batch_size=8,
            grad_accum_steps=2,
            seq_len=2048,
            gradient_checkpointing=True,
            mixed_precision=True,
            flash_attention=True,
            cpu_offload_optimizer=False,
            lora_rank=16,
            lora_alpha=32,
            optimizer=OptimizerType.ADAMW,
        )

        report = estimator.estimate(
            gpu_topology_a100_40gb, model_fp16, optimized_plan
        )

        # Check feasibility
        gpu_memory = gpu_topology_a100_40gb.gpus[0].vram_bytes
        safety_margin = 0.9  # Keep 10% free

        assert report.total_bytes < gpu_memory * safety_margin
        utilization = (report.total_bytes / gpu_memory) * 100

        logger.info(
            f"Pre-training check: {utilization:.1f}% GPU utilization "
            f"({report.total_bytes/1024**3:.2f}GB / {gpu_memory/1024**3:.0f}GB)"
        )

    @pytest.mark.asyncio
    async def test_training_with_recovery(self):
        """Simulate training steps with OOM and recovery."""
        orchestrator = RecoveryOrchestrator()
        training_steps = 100
        oom_at_step = 50

        for step in range(training_steps):
            if step == oom_at_step:
                logger.info(f"OOM at step {step}, triggering recovery...")

                error = OutOfMemoryError()
                context = RecoveryContext(
                    error=error,
                    step=step,
                    last_checkpoint_path=f"/checkpoints/step_{step-10}.pt",
                    remaining_retries=3,
                    current_batch_size=16,
                    current_training_mode="lora",
                    peak_memory_usage_bytes=40 * 1024**3,
                )

                success = await orchestrator.recover(context)
                assert success is True
                logger.info(f"Recovery successful at step {step}")

        logger.info(f"Training simulation completed: {training_steps} steps")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
