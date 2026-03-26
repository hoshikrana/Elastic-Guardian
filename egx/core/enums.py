"""
EGX Core Enums — Layer 1.

Every enum is string-typed for JSON serialization compatibility.
Law 12: Public surface frozen from v0.1.
"""

from __future__ import annotations

from enum import Enum


class DeviceType(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"


class HardwareTier(str, Enum):
    LAPTOP = "laptop"  # ≤ 12GB
    WORKSTATION = "workstation"  # ≤ 48GB
    PROSUMER = "prosumer"  # ≤ 80GB
    DATACENTER = "datacenter"
    CLUSTER = "cluster"


class InterconnectType(str, Enum):
    NONE = "none"
    PCIE = "pcie"
    NVLINK = "nvlink"
    INFINIBAND = "infiniband"


class ThermalState(str, Enum):
    NOMINAL = "nominal"
    WARM = "warm"
    THROTTLING = "throttling"
    CRITICAL = "critical"


class RecoveryState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    THROTTLED = "throttled"
    SUSPENDED = "suspended"
    RECONFIGURING = "reconfiguring"
    CHECKPOINTING = "checkpointing"
    ESCALATED = "escalated"
    ABORTED = "aborted"


class MemoryTier(str, Enum):
    VRAM = "vram"
    RAM = "ram"
    NVME = "nvme"
    REMOTE = "remote"


class DType(str, Enum):
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    INT8 = "int8"
    INT4 = "int4"

    def byte_size(self) -> int:
        mapping = {
            DType.FP32: 4,
            DType.FP16: 2,
            DType.BF16: 2,
            DType.INT8: 1,
            DType.INT4: 1,  # Technically 0.5 but for alignment usually 1
        }
        return mapping[self]


class MemoryPressureLevel(str, Enum):
    NOMINAL = "nominal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class TrainingMode(str, Enum):
    FULL_FINETUNE = "full"
    LORA = "lora"
    LORA_PLUS = "lora_plus"
    DORA = "dora"
    QLORA = "qlora"
    PREFIX = "prefix"
    ADAPTER = "adapter"
    PHANTOM = "phantom"

    def uses_peft(self) -> bool:
        """Returns True if this training mode uses parameter-efficient fine-tuning."""
        return self in (
            TrainingMode.LORA,
            TrainingMode.LORA_PLUS,
            TrainingMode.DORA,
            TrainingMode.QLORA,
            TrainingMode.PREFIX,
            TrainingMode.ADAPTER,
        )

    def weight_dtype(self, base_dtype: "DType") -> "DType":
        """Returns the storage DType for model weights under this training mode."""
        if self == TrainingMode.QLORA:
            return DType.INT4
        return base_dtype


class ParallelStrategy(str, Enum):
    NONE = "none"
    DATA = "data"
    FSDP = "fsdp"
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    HYBRID_3D = "hybrid_3d"


class OptimizerType(str, Enum):
    ADAMW = "adamw"
    ADAM_8BIT = "adam_8bit"
    ADAFACTOR = "adafactor"
    SGD = "sgd"

    def bytes_per_param(self) -> int:
        """Returns the optimizer state bytes per trainable parameter.
        
        AdamW keeps 2 states (m, v) in FP32 = 8 bytes per param.
        Adam 8-bit keeps 2 states in INT8 = 2 bytes per param.
        Adafactor keeps ~1 state = 4 bytes per param.
        SGD with momentum keeps 1 state = 4 bytes per param.
        """
        mapping = {
            OptimizerType.ADAMW: 8,
            OptimizerType.ADAM_8BIT: 2,
            OptimizerType.ADAFACTOR: 4,
            OptimizerType.SGD: 4,
        }
        return mapping.get(self, 8)


class RecoveryAction(str, Enum):
    RETRY = "retry"
    HALVE_BATCH = "halve_batch"
    DOWNGRADE_STRATEGY = "downgrade_strategy"
    RELOAD_CHECKPOINT = "reload_checkpoint"
    RESTART_PROCESS = "restart_process"
    ABORT = "abort"
    SKIP_BATCH = "skip_batch"


class CheckpointStrategy(str, Enum):
    LOSS_BASED = "loss_based"
    TIME_BASED = "time_based"
    STEP_BASED = "step_based"
    ADAPTIVE = "adaptive"


class EstimationMethod(str, Enum):
    ANALYTICAL = "analytical"
    DRY_RUN = "dryrun"
    HYBRID = "hybrid"
    ML_BASED = "ml_based"


class ArchType(str, Enum):
    TRANSFORMER = "transformer"
    LLAMA = "llama"
    MISTRAL = "mistral"
    FALCON = "falcon"
    PHANTOM = "phantom"
