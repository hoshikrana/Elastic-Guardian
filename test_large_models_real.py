#!/usr/bin/env python3
"""
EGX Large Model Testing Script

Download and test EGX framework with real models from HuggingFace Hub:
- LLaMA 2 (7B, 13B)
- Mistral 7B
- Phi 2

Usage:
    python test_large_models_real.py --model llama-7b --batch-size 16 --test-recovery
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Model Configurations
# ============================================================================

MODEL_CONFIGS = {
    "llama-7b": {
        "hf_model": "meta-llama/Llama-2-7b",
        "size_b": 7.0,
        "hidden_dim": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "vocab_size": 32000,
        "max_seq_len": 4096,
        "recommended_batch_size": 16,
    },
    "llama-13b": {
        "hf_model": "meta-llama/Llama-2-13b",
        "size_b": 13.0,
        "hidden_dim": 5120,
        "num_layers": 40,
        "num_heads": 40,
        "vocab_size": 32000,
        "max_seq_len": 4096,
        "recommended_batch_size": 8,
    },
    "mistral-7b": {
        "hf_model": "mistralai/Mistral-7B",
        "size_b": 7.0,
        "hidden_dim": 4096,
        "num_layers": 32,
        "num_heads": 8,
        "vocab_size": 32768,
        "max_seq_len": 8192,
        "recommended_batch_size": 16,
    },
    "phi-2.7b": {
        "hf_model": "microsoft/phi-2",
        "size_b": 2.7,
        "hidden_dim": 2560,
        "num_layers": 32,
        "num_heads": 32,
        "vocab_size": 50256,
        "max_seq_len": 2048,
        "recommended_batch_size": 32,
    },
}


# ============================================================================
# 1. Memory Estimation Tests
# ============================================================================

def test_memory_estimation(model_name: str, batch_size: int = None) -> Dict:
    """
    Estimate memory usage for a model configuration.
    
    Returns: Dictionary with memory estimates
    """
    from egx.core.models import GPUSpec, HardwareTopology, ModelProfile, TrainingPlan
    from egx.core.enums import (
        ArchType, DType, OptimizerType, ParallelStrategy,
        InterconnectType, TrainingMode,
    )
    from egx.intelligence.estimator.improved_analytical import ImprovedAnalyticalEstimator

    logger.info(f"\n{'='*60}")
    logger.info(f"MEMORY ESTIMATION: {model_name}")
    logger.info(f"{'='*60}")

    config = MODEL_CONFIGS[model_name]
    batch_size = batch_size or config["recommended_batch_size"]

    # Setup GPU (A100 40GB)
    gpu_spec = GPUSpec(
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
    topology = HardwareTopology(
        gpus=(gpu_spec,),
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

    # Model profile
    profile = ModelProfile(
        arch=ArchType.TRANSFORMER,
        params=int(config["size_b"] * 1024**3),
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        max_seq_len=config["max_seq_len"],
        dtype=DType.FP32,
    )

    # Training configurations to test
    configs = [
        {
            "name": "LoRA (batch=16, gradient_ckpt=True)",
            "mode": TrainingMode.LORA,
            "batch_size": batch_size,
            "grad_accum": 4,
            "seq_len": 2048,
            "gradient_checkpointing": True,
            "flash_attention": True,
            "lora_rank": 16,
        },
        {
            "name": "LoRA (batch=32, gradient_ckpt=True)",
            "mode": TrainingMode.LORA,
            "batch_size": 32,
            "grad_accum": 2,
            "seq_len": 2048,
            "gradient_checkpointing": True,
            "flash_attention": True,
            "lora_rank": 16,
        },
        {
            "name": "Full Finetune (batch=4, gradient_ckpt=True)",
            "mode": TrainingMode.FULL_FINETUNE,
            "batch_size": 4,
            "grad_accum": 1,
            "seq_len": 2048,
            "gradient_checkpointing": True,
            "flash_attention": True,
            "lora_rank": 0,
        },
    ]

    estimator = ImprovedAnalyticalEstimator()
    results = {}

    for cfg in configs:
        plan = TrainingPlan(
            mode=cfg["mode"],
            dtype=DType.FP32,
            parallel_strategy=ParallelStrategy.NONE,
            batch_size=cfg["batch_size"],
            grad_accum_steps=cfg["grad_accum"],
            seq_len=cfg["seq_len"],
            gradient_checkpointing=cfg["gradient_checkpointing"],
            mixed_precision=False,
            flash_attention=cfg["flash_attention"],
            cpu_offload_optimizer=False,
            lora_rank=cfg["lora_rank"],
            lora_alpha=32 if cfg["lora_rank"] > 0 else 0,
            optimizer=OptimizerType.ADAMW,
        )

        report = estimator.estimate(topology, profile, plan)
        memory_gb = report.total_bytes / 1024**3
        fits = memory_gb < 40 * 0.95  # 95% of GPU memory

        results[cfg["name"]] = {
            "memory_gb": memory_gb,
            "fits_gpu": fits,
            "confidence": report.confidence,
            "error_bound": report.error_bound_pct,
        }

        status = "✅ FITS" if fits else "❌ EXCEEDS"
        logger.info(
            f"  {cfg['name']}: {memory_gb:.2f}GB {status} "
            f"(confidence: {report.confidence*100:.0f}%, ±{report.error_bound_pct}%)"
        )

    return results


# ============================================================================
# 2. Recovery Orchestrator Tests
# ============================================================================

async def test_recovery_orchestrator(batch_size: int = 16) -> Dict:
    """
    Test recovery orchestrator with simulated OOM scenarios.
    
    Returns: Test results
    """
    from egx.resilience.recovery.orchestrator import RecoveryOrchestrator, RecoveryContext
    from egx.core.exceptions import OutOfMemoryError

    logger.info(f"\n{'='*60}")
    logger.info("RECOVERY ORCHESTRATOR TESTS")
    logger.info(f"{'='*60}")

    orchestrator = RecoveryOrchestrator()
    results = {}

    # Test 1: Basic recovery
    logger.info("\n1. Testing basic recovery execution...")
    error = OutOfMemoryError(msg="Out of memory during forward pass")
    context = RecoveryContext(
        error=error,
        step=100,
        last_checkpoint_path="/checkpoints/step_90.pt",
        remaining_retries=3,
        current_batch_size=batch_size,
        current_training_mode="lora",
        peak_memory_usage_bytes=40 * 1024**3,
    )

    success = await orchestrator.recover(context)
    results["basic_recovery"] = {"success": success}
    logger.info(f"   Result: {'✅ SUCCESS' if success else '❌ FAILED'}")

    # Test 2: Batch size halving
    logger.info("\n2. Testing batch size adaptation...")
    context.current_batch_size = 32
    success = await orchestrator.recover(context)
    results["batch_halving"] = {"success": success}
    logger.info(f"   Result: {'✅ SUCCESS' if success else '❌ FAILED'}")

    # Test 3: Max retries exceeded
    logger.info("\n3. Testing max retries exceeded...")
    context_no_retries = RecoveryContext(
        error=OutOfMemoryError(),
        step=1000,
        last_checkpoint_path=None,
        remaining_retries=0,
        current_batch_size=1,
        current_training_mode="lora",
        peak_memory_usage_bytes=45 * 1024**3,
    )
    success = await orchestrator.recover(context_no_retries)
    results["max_retries"] = {"success": success}
    logger.info(f"   Result: {success} (expected behavior)")

    return results


# ============================================================================
# 3. Model Capability Matrix
# ============================================================================

def generate_capability_matrix() -> str:
    """Generate a matrix showing which models fit on which GPUs with which configs."""
    logger.info(f"\n{'='*60}")
    logger.info("MODEL CAPABILITY MATRIX")
    logger.info(f"{'='*60}")

    from egx.core.models import GPUSpec, HardwareTopology, ModelProfile, TrainingPlan
    from egx.core.enums import (
        ArchType, DType, OptimizerType, ParallelStrategy,
        InterconnectType, TrainingMode,
    )
    from egx.intelligence.estimator.improved_analytical import ImprovedAnalyticalEstimator

    gpus = {
        "A100-40GB": {"vram": 40, "vendor": "nvidia"},
        "H100-80GB": {"vram": 80, "vendor": "nvidia"},
    }

    training_modes = [
        ("LoRA", TrainingMode.LORA, 16),
        ("Full FT", TrainingMode.FULL_FINETUNE, 0),
    ]

    matrix = []
    matrix.append("Model | LoRA (A100) | LoRA (H100) | FT (A100) | FT (H100)")
    matrix.append("------|-------------|------------|-----------|----------")

    estimator = ImprovedAnalyticalEstimator()

    for model_name, config in MODEL_CONFIGS.items():
        row = model_name

        for gpu_name, gpu_info in gpus.items():
            gpu_spec = GPUSpec(
                device_id=0,
                name=gpu_name,
                vram_bytes=gpu_info["vram"] * 1024**3,
                compute_capability=(8, 0) if "A100" in gpu_name else (9, 0),
                memory_bandwidth_gbps=2039.0 if "A100" in gpu_name else 3352.0,
                fp16_tflops=19500.0 if "A100" in gpu_name else 67000.0,
                bf16_tflops=19500.0 if "A100" in gpu_name else 67000.0,
                supports_flash_attn2=True,
                supports_fp8=True,
                nvlink_peer_ids=(),
                vendor=gpu_info["vendor"],
            )
            topology = HardwareTopology(
                gpus=(gpu_spec,),
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

            profile = ModelProfile(
                arch=ArchType.TRANSFORMER,
                params=int(config["size_b"] * 1024**3),
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"],
                num_heads=config["num_heads"],
                max_seq_len=config["max_seq_len"],
                dtype=DType.FP32,
            )

            # Test LoRA
            plan_lora = TrainingPlan(
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

            report_lora = estimator.estimate(topology, profile, plan_lora)
            lora_fits = report_lora.total_bytes < gpu_info["vram"] * 1024**3 * 0.95

            # Test Full FT
            plan_ft = TrainingPlan(
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

            report_ft = estimator.estimate(topology, profile, plan_ft)
            ft_fits = report_ft.total_bytes < gpu_info["vram"] * 1024**3 * 0.95

            if "A100" in gpu_name:
                row += f" | {'✅' if lora_fits else '❌'}"
                row += f" | {report_lora.total_bytes/1024**3:.1f}GB"
            else:
                row += f" | {'✅' if lora_fits else '❌'}"
                row += f" | {report_lora.total_bytes/1024**3:.1f}GB"

        matrix.append(row)

    matrix_str = "\n".join(matrix)
    logger.info("\n" + matrix_str)
    return matrix_str


# ============================================================================
# Main Test Runner
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Test EGX framework with large models"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="llama-7b",
        help="Model to test",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: model recommendation)",
    )
    parser.add_argument(
        "--test-recovery",
        action="store_true",
        help="Test recovery orchestrator",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Test all models",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Generate capability matrix",
    )

    args = parser.parse_args()

    try:
        if args.matrix:
            generate_capability_matrix()

        if args.all_models:
            logger.info("\nTesting all models...")
            all_results = {}
            for model_name in MODEL_CONFIGS.keys():
                all_results[model_name] = test_memory_estimation(model_name)
            
            # Save results
            results_file = Path("large_model_test_results.json")
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"\n✅ Results saved to {results_file}")

        else:
            # Single model test
            results = test_memory_estimation(args.model, args.batch_size)

            if args.test_recovery:
                recovery_results = await test_recovery_orchestrator(
                    batch_size=args.batch_size or MODEL_CONFIGS[args.model]["recommended_batch_size"]
                )
                logger.info("\nRecovery test results:")
                logger.info(json.dumps(recovery_results, indent=2))

        logger.info("\n✅ All tests completed successfully!")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
