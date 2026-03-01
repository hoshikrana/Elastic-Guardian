"""
EGX core enumerations.

Rules (Law 1, Law 6):
  - stdlib only — no torch, no third-party imports
  - all enums inherit str so values serialize to JSON without .value
  - every tier/level enum exposes ordering helpers
"""

from __future__ import annotations

from enum import Enum


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

class DeviceType(str, Enum):
    """Physical compute device category."""
    CUDA = "cuda"         # NVIDIA GPU via CUDA
    CPU  = "cpu"          # CPU-only (no accelerator)
    MPS  = "mps"          # Apple Silicon unified memory (Metal Performance Shaders)
    ROCM = "rocm"         # AMD GPU via ROCm / HIP


class HardwareTier(str, Enum):
    """
    Auto-computed from total VRAM. Used by strategy selector to short-circuit
    impossible strategies before scoring.

    Boundaries (inclusive upper, GiB):
      LAPTOP      <= 12    consumer mobile / desktop low-end
      WORKSTATION <= 48    RTX 3090/4090, A6000
      PROSUMER    <= 96    A100 40/80, H100 PCIe
      DATACENTER  <= 640   multi-A100 single node
      CLUSTER     > 640    multi-node
    """
    LAPTOP      = "laptop"
    WORKSTATION = "workstation"
    PROSUMER    = "prosumer"
    DATACENTER  = "datacenter"
    CLUSTER     = "cluster"

    @classmethod
    def from_vram_bytes(cls, vram_bytes: int) -> "HardwareTier":
        """Derive tier from total VRAM across all GPUs. Input is int bytes."""
        if not isinstance(vram_bytes, int) or isinstance(vram_bytes, bool):
            raise TypeError(f"vram_bytes must be int, got {type(vram_bytes)}")
        gib = vram_bytes / 1_073_741_824
        if gib <= 12:
            return cls.LAPTOP
        if gib <= 48:
            return cls.WORKSTATION
        if gib <= 96:
            return cls.PROSUMER
        if gib <= 640:
            return cls.DATACENTER
        return cls.CLUSTER


class InterconnectType(str, Enum):
    """GPU-to-GPU interconnect detected at probe time."""
    PCIE       = "pcie"
    NVLINK     = "nvlink"
    INFINIBAND = "infiniband"
    NONE       = "none"   # single GPU


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class MemoryTier(str, Enum):
    """
    Storage tier for tensor placement. Ordered by access latency (fastest first).
    Allocation planner decides where each tensor lives.
    """
    VRAM   = "vram"    # GPU HBM — fastest
    RAM    = "ram"     # CPU DRAM — ~10x slower than VRAM
    NVME   = "nvme"    # NVMe SSD — ~100x slower than VRAM
    REMOTE = "remote"  # network-attached — cluster mode only

    def rank(self) -> int:
        """0 = fastest. Used for eviction ordering (evict highest rank first)."""
        return {
            MemoryTier.VRAM:   0,
            MemoryTier.RAM:    1,
            MemoryTier.NVME:   2,
            MemoryTier.REMOTE: 3,
        }[self]


class DType(str, Enum):
    """
    Weight data types. byte_size() returns bytes per element.
    INT4 returns 0.5 because two values are packed per byte.
    """
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    INT8 = "int8"
    INT4 = "int4"

    def byte_size(self) -> float:
        """Bytes per element. Use int(params * dtype.byte_size()) in formulas."""
        return {
            DType.FP32: 4.0,
            DType.FP16: 2.0,
            DType.BF16: 2.0,
            DType.INT8: 1.0,
            DType.INT4: 0.5,
        }[self]


class MemoryPressureLevel(str, Enum):
    """
    Five-threshold VRAM pressure model.
    Thresholds match config/default.yaml:orchestration.pressure.thresholds

    GREEN     < 72%   nominal, safe to grow batch
    YELLOW    72-85%  light pressure, hold batch
    ORANGE    85-92%  moderate, reduce batch x0.85
    RED       92-97%  high, swap optimizer states to RAM
    EMERGENCY > 97%   OOM imminent, halve batch immediately
    """
    GREEN     = "green"
    YELLOW    = "yellow"
    ORANGE    = "orange"
    RED       = "red"
    EMERGENCY = "emergency"

    def rank(self) -> int:
        """0 = lowest pressure. Used for comparison and sorting."""
        _order = ["green", "yellow", "orange", "red", "emergency"]
        return _order.index(self.value)

    def is_above(self, other: "MemoryPressureLevel") -> bool:
        """True if self represents more pressure than other."""
        return self.rank() > other.rank()

    @classmethod
    def from_fraction(cls, used: int, total: int) -> "MemoryPressureLevel":
        """
        Derive pressure level from used/total VRAM bytes.
        Both arguments must be positive int.
        """
        if total <= 0:
            raise ValueError(f"total must be positive, got {total}")
        frac = used / total
        if frac < 0.72:
            return cls.GREEN
        if frac < 0.85:
            return cls.YELLOW
        if frac < 0.92:
            return cls.ORANGE
        if frac < 0.97:
            return cls.RED
        return cls.EMERGENCY


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainingMode(str, Enum):
    """
    Fine-tuning strategy. Strategy selector traverses in definition order
    (highest quality first) and picks the first strategy that fits in VRAM.
    """
    FULL_FINETUNE = "full_finetune"
    LORA_PLUS     = "lora_plus"
    LORA          = "lora"
    DORA          = "dora"
    QLORA         = "qlora"

    def uses_peft(self) -> bool:
        return self != TrainingMode.FULL_FINETUNE

    def uses_quantization(self) -> bool:
        return self == TrainingMode.QLORA

    def safety_threshold(self) -> float:
        """
        Max VRAM fraction allowed for this strategy.
        Per-strategy, not a flat 80% — critical architectural decision.
        Source: architecture Chapter 6.2
        """
        return {
            TrainingMode.FULL_FINETUNE: 0.72,
            TrainingMode.LORA_PLUS:     0.80,
            TrainingMode.LORA:          0.80,
            TrainingMode.DORA:          0.82,
            TrainingMode.QLORA:         0.90,
        }[self]


class ParallelStrategy(str, Enum):
    """Multi-GPU parallelism. Selected by parallel_advisor.py."""
    NONE      = "none"
    DATA      = "data"
    FSDP      = "fsdp"
    TENSOR    = "tensor"
    PIPELINE  = "pipeline"
    HYBRID_3D = "hybrid_3d"


class OptimizerType(str, Enum):
    """
    Optimizer. bytes_per_param() determines optimizer memory in formulas.
    Critical: for LoRA/QLoRA, apply to P_lora (trainable params), NOT P (all params).
    """
    ADAMW      = "adamw"
    ADAM_8BIT  = "adam_8bit"
    ADAFACTOR  = "adafactor"
    SGD        = "sgd"

    def bytes_per_param(self) -> float:
        """Memory cost per trainable parameter."""
        return {
            OptimizerType.ADAMW:     8.0,   # 2x FP32 moments
            OptimizerType.ADAM_8BIT: 2.0,   # 8-bit quantized moments
            OptimizerType.ADAFACTOR: 4.0,   # factored second moment
            OptimizerType.SGD:       4.0,   # momentum buffer
        }[self]


class SchedulerType(str, Enum):
    """LR scheduler. Stored in TrainingPlan for reproducibility."""
    COSINE          = "cosine"
    LINEAR          = "linear"
    CONSTANT        = "constant"
    COSINE_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL      = "polynomial"


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------

class FailureType(str, Enum):
    """
    All failure modes EGX handles. Each maps to a typed exception
    and a recovery chain in RecoveryFSM.
    """
    OOM                = "oom"
    NAN_LOSS           = "nan_loss"
    INF_GRADIENT       = "inf_gradient"
    DEADLOCK           = "deadlock"
    DRIVER_CRASH       = "driver_crash"
    THERMAL_THROTTLE   = "thermal_throttle"
    NVME_TIMEOUT       = "nvme_timeout"
    NCCL_TIMEOUT       = "nccl_timeout"
    CHECKPOINT_CORRUPT = "checkpoint_corrupt"


class RecoveryAction(str, Enum):
    """
    Action the training kernel takes when RecoveryFSM makes a decision.
    Ordered from least to most disruptive.
    """
    RETRY              = "retry"
    HALVE_BATCH        = "halve_batch"
    DOWNGRADE_STRATEGY = "downgrade_strategy"
    RELOAD_CHECKPOINT  = "reload_checkpoint"
    RESTART_PROCESS    = "restart_process"
    ABORT              = "abort"


class RecoveryState(str, Enum):
    """Internal states of the RecoveryFSM."""
    HEALTHY    = "healthy"
    DEGRADED   = "degraded"
    RECOVERING = "recovering"
    ESCALATED  = "escalated"
    ABORTED    = "aborted"


# ---------------------------------------------------------------------------
# Checkpoint / Estimation / Architecture
# ---------------------------------------------------------------------------

class CheckpointStrategy(str, Enum):
    """How CheckpointManager decides when to save."""
    LOSS_BASED = "loss_based"
    TIME_BASED = "time_based"
    STEP_BASED = "step_based"
    ADAPTIVE   = "adaptive"


class EstimationMethod(str, Enum):
    """Which pipeline produced a MemoryReport."""
    ANALYTICAL   = "analytical"
    DRYRUN       = "dryrun"
    HYBRID       = "hybrid"
    ML_CORRECTED = "ml_corrected"


class ArchType(str, Enum):
    """
    Known model architecture families for registry.py and introspector.py.
    UNKNOWN triggers broad heuristic target detection in peft/injector.py.
    """
    LLAMA   = "llama"
    MISTRAL = "mistral"
    FALCON  = "falcon"
    GPT2    = "gpt2"
    GPTJ    = "gptj"
    GPTNEOX = "gptneox"
    T5      = "t5"
    BERT    = "bert"
    ROBERTA = "roberta"
    UNKNOWN = "unknown"