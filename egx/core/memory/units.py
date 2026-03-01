"""
Memory unit conversion — pure functions.

All inputs and outputs are int bytes (Law 10).
Float GB is only accepted on the way *in* from user input,
and only with an explicit conversion function that makes the rounding visible.
Validator is injected, never a global import.

Design principle: this module contains FORMULAS, not constants.
Constants (activation multipliers, grad_ckpt factors, overhead ratios)
are owned by intelligence/estimator/ which derives them from dry-run
measurements on real hardware at runtime.
"""

from __future__ import annotations

import math

from egx.core.constants import KB, MB, GB, TB
from egx.core.memory.validators import MemoryValidator, get_default_validator

# ---------------------------------------------------------------------------
# From user-facing units -> bytes (int)
# These are the only places float GB is allowed to touch EGX internals.
# ---------------------------------------------------------------------------

def gib_to_bytes(gib: float, validator: MemoryValidator | None = None) -> int:
    """
    Convert GiB (float) to bytes (int). Rounds up to avoid underestimation.

    Args:
        gib:       Size in gibibytes (1 GiB = 1024^3 bytes).
        validator: Optional validator. Defaults to StandardMemoryValidator.

    Returns:
        int bytes, rounded up.
    """
    result = math.ceil(gib * GB)
    v = validator or get_default_validator()
    return v.validate(result, "gib_to_bytes result")


def gb_to_bytes(gb: float, validator: MemoryValidator | None = None) -> int:
    """
    Convert GB (float, SI: 10^9) to bytes (int).
    Note: 1 GB = 1_000_000_000 bytes. Not the same as GiB.
    """
    result = math.ceil(gb * 1_000_000_000)
    v = validator or get_default_validator()
    return v.validate(result, "gb_to_bytes result")


def mib_to_bytes(mib: float) -> int:
    """Convert MiB to bytes."""
    return math.ceil(mib * MB)


def kib_to_bytes(kib: float) -> int:
    """Convert KiB to bytes."""
    return math.ceil(kib * KB)


# ---------------------------------------------------------------------------
# From bytes -> display units (float, for human output only)
# Never use these values in arithmetic — always use raw bytes.
# ---------------------------------------------------------------------------

def bytes_to_gib(b: int) -> float:
    """Bytes to GiB. For display only."""
    return b / GB


def bytes_to_gb(b: int) -> float:
    """Bytes to GB (SI). For display only."""
    return b / 1_000_000_000


def bytes_to_mib(b: int) -> float:
    """Bytes to MiB. For display only."""
    return b / MB


# ---------------------------------------------------------------------------
# Parameter count -> memory (int bytes)
# Pure formulas. No strategy-specific constants.
# ---------------------------------------------------------------------------

def params_to_bytes(param_count: int, bytes_per_element: float) -> int:
    """
    Convert parameter count to memory bytes.

    Args:
        param_count:       Number of parameters (int).
        bytes_per_element: Bytes per param. e.g:
                             FP32 = 4.0, BF16/FP16 = 2.0,
                             INT8 = 1.0, INT4 = 0.5
                           Supplied by caller from DType.byte_size().

    Returns:
        int bytes, rounded up (conservative).
    """
    if isinstance(param_count, bool):
        raise TypeError(f"param_count must be int, got bool: {param_count!r}")
    if not isinstance(param_count, int):
        raise TypeError(f"param_count must be int, got {type(param_count).__name__}")
    if param_count < 0:
        raise ValueError(f"param_count must be >= 0, got {param_count}")
    if bytes_per_element <= 0:
        raise ValueError(f"bytes_per_element must be > 0, got {bytes_per_element}")

    return math.ceil(param_count * bytes_per_element)


def optimizer_memory_bytes(
    trainable_param_count: int,
    bytes_per_param: float,
) -> int:
    """
    Optimizer state memory.

    CRITICAL: For LoRA/QLoRA, trainable_param_count is the adapter parameter
    count (P_lora), NOT the full model parameter count (P).
    Using P for LoRA optimizer memory is wrong — for a 7B model with rank=16,
    adapter params are ~21M, not 7B. Using P overstates by ~333x.

    The caller (estimator) is responsible for passing the correct count.
    This function is a formula, not a policy.

    Args:
        trainable_param_count: Number of trainable parameters.
        bytes_per_param:       Optimizer memory per trainable param.
                               Supplied by caller from OptimizerType.bytes_per_param().
    """
    return params_to_bytes(trainable_param_count, bytes_per_param)


def activation_memory_bytes(
    batch_size:            int,
    seq_len:               int,
    hidden_dim:            int,
    num_layers:            int,
    bytes_per_el:          float,
    activation_multiplier: float,
    grad_ckpt_factor:      float = 1.0,
) -> int:
    """
    Activation memory formula for transformer models.

    This is a pure formula — no hardcoded constants.
    All multipliers are supplied by the caller.

    The estimator layer (intelligence/estimator/) owns the multiplier values.
    It derives them from:
      - Dry-run: allocates a real forward+backward pass on actual hardware
                 and measures peak memory directly from torch.cuda.memory_stats()
      - Analytical fallback: uses architecture-justified defaults when dry-run
                             is not yet available (e.g. first call, cold start)

    Why constants are NOT here:
        The activation multiplier depends on architecture (standard vs MLA vs GQA),
        CUDA version, attention implementation (FlashAttention vs SDPA vs naive),
        training mode (full FT vs LoRA vs QLoRA), and hardware (A100 vs H100 vs
        consumer GPUs with different memory bandwidth). Baking numbers here
        would make the estimator wrong on any environment we haven't pre-measured.
        The dry-run estimator measures actual allocations on the actual hardware
        at runtime. The analytical estimator uses defaults that the calibrator
        refines over time. Neither lives here.

    Formula:
        raw = batch_size * seq_len * hidden_dim * num_layers
                        * activation_multiplier * bytes_per_el
                        * grad_ckpt_factor
        return ceil(raw)

    Args:
        batch_size:            Per-GPU batch size (int, not bool).
        seq_len:               Sequence length.
        hidden_dim:            Model hidden dimension.
        num_layers:            Number of transformer layers.
        bytes_per_el:          Bytes per activation element. 2.0 for BF16.
        activation_multiplier: Dimensionless scaling factor.
                               Owned and supplied by intelligence/estimator/.
        grad_ckpt_factor:      1.0 = no checkpointing (default).
                               0.0 < factor < 1.0 = checkpointing active.
                               Owned and supplied by intelligence/estimator/.

    Returns:
        int bytes.
    """
    if any(isinstance(v, bool) for v in (batch_size, seq_len, hidden_dim, num_layers)):
        raise TypeError("All dimension arguments must be int, not bool")
    if activation_multiplier <= 0:
        raise ValueError(f"activation_multiplier must be > 0, got {activation_multiplier}")
    if not (0.0 < grad_ckpt_factor <= 1.0):
        raise ValueError(f"grad_ckpt_factor must be in (0, 1], got {grad_ckpt_factor}")

    raw = (
        batch_size
        * seq_len
        * hidden_dim
        * num_layers
        * activation_multiplier
        * bytes_per_el
        * grad_ckpt_factor
    )
    return math.ceil(raw)
