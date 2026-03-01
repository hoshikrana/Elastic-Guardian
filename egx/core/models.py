"""
EGX core dataclasses — the public contracts between all layers.

Rules:
  - frozen=True, slots=True on every dataclass (Law 5)
  - No torch imports — stdlib + egx.core only
  - All memory fields: int bytes (Law 10) — never float GB
  - Every cross-layer boundary passes only these types or typed exceptions
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from egx.core.enums import (
    ArchType,
    DeviceType,
    DType,
    EstimationMethod,
    FailureType,
    HardwareTier,
    InterconnectType,
    MemoryPressureLevel,
    MemoryTier,
    OptimizerType,
    ParallelStrategy,
    RecoveryAction,
    RecoveryState,
    SchedulerType,
    TrainingMode,
)

# ---------------------------------------------------------------------------
# Hardware contracts (Layer 2 -> Layer 3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GPUSpec:
    """
    Specification for a single GPU. Produced by gpu_probe.py.
    All memory values: int bytes (Law 10).
    """
    gpu_id:            int
    name:              str
    device_type:       DeviceType
    vram_total:        int          # bytes
    vram_free:         int          # bytes at probe time
    compute_major:     int
    compute_minor:     int
    nvlink_peer_ids:   tuple[int, ...]  = field(default_factory=tuple)
    pcie_bus_id:       str              = ""
    temp_celsius:      float            = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.vram_total, bool) or isinstance(self.vram_free, bool):
            raise TypeError("vram fields must be int, not bool")
        if self.vram_total < 0:
            raise ValueError(f"vram_total must be >= 0, got {self.vram_total}")
        if self.vram_free < 0:
            raise ValueError(f"vram_free must be >= 0, got {self.vram_free}")
        if self.vram_free > self.vram_total:
            raise ValueError(
                f"vram_free ({self.vram_free}) > vram_total ({self.vram_total})"
            )

    @property
    def vram_used(self) -> int:
        return self.vram_total - self.vram_free

    @property
    def tier(self) -> HardwareTier:
        return HardwareTier.from_vram_bytes(self.vram_total)

    @property
    def compute_capability(self) -> str:
        return f"{self.compute_major}.{self.compute_minor}"

    @property
    def supports_bf16(self) -> bool:
        """BF16 requires compute capability >= 8.0 (Ampere+)."""
        return self.compute_major >= 8

    @property
    def supports_flash_attention(self) -> bool:
        """FlashAttention2 requires compute capability >= 8.0."""
        return self.compute_major >= 8


@dataclass(frozen=True, slots=True)
class HardwareTopology:
    """
    Complete hardware description. Produced by topology_builder.py.
    Passed from Layer 2 -> Layer 3. Immutable.
    All memory values: int bytes (Law 10).
    """
    gpus:                tuple[GPUSpec, ...]
    cpu_ram_total:        int                   # bytes
    cpu_ram_free:         int                   # bytes
    nvme_path:            str                   = ""
    nvme_capacity:        int                   = 0    # bytes
    nvme_read_bw:         int                   = 0    # bytes/s
    nvme_write_bw:        int                   = 0    # bytes/s
    pcie_bandwidth:       int                   = 0    # bytes/s (host<->device)
    nvlink_bandwidth:     int                   = 0    # bytes/s (device<->device)
    interconnect_type:    InterconnectType      = InterconnectType.NONE

    def __post_init__(self) -> None:
        if not self.gpus:
            raise ValueError("HardwareTopology requires at least one GPUSpec")
        for mem in (self.cpu_ram_total, self.cpu_ram_free):
            if isinstance(mem, bool):
                raise TypeError("RAM fields must be int, not bool")
            if mem < 0:
                raise ValueError(f"RAM field must be >= 0, got {mem}")

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def total_vram(self) -> int:
        return sum(g.vram_total for g in self.gpus)

    @property
    def total_vram_free(self) -> int:
        return sum(g.vram_free for g in self.gpus)

    @property
    def tier(self) -> HardwareTier:
        return HardwareTier.from_vram_bytes(self.total_vram)

    @property
    def has_nvlink(self) -> bool:
        return self.interconnect_type == InterconnectType.NVLINK

    @property
    def has_nvme(self) -> bool:
        return self.nvme_capacity > 0

    @property
    def supports_bf16(self) -> bool:
        return all(g.supports_bf16 for g in self.gpus)


# ---------------------------------------------------------------------------
# Model contract (Layer 5 models/ -> Layer 3 intelligence/)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ModelProfile:
    """
    Everything EGX needs to know about a model to plan training.
    Produced by models/introspector.py. Consumed by intelligence layer.
    """
    arch_type:          ArchType
    total_params:       int          # total parameter count
    trainable_params:   int          # params that require grad (pre-PEFT)
    hidden_dim:         int
    num_layers:         int
    num_heads:          int
    vocab_size:         int
    max_seq_len:        int
    weight_dtype:       DType        = DType.FP32
    default_lora_rank:  int          = 16
    lora_target_modules: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.total_params <= 0:
            raise ValueError(f"total_params must be > 0, got {self.total_params}")
        if self.trainable_params > self.total_params:
            raise ValueError(
                f"trainable_params ({self.trainable_params}) > "
                f"total_params ({self.total_params})"
            )
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")

    @property
    def param_size_bytes(self) -> int:
        """Total weight memory at the model's native dtype. int bytes."""
        return int(self.total_params * self.weight_dtype.byte_size())

    @property
    def param_size_bf16_bytes(self) -> int:
        """Weight memory if cast to BF16 (used in most fine-tuning)."""
        return int(self.total_params * DType.BF16.byte_size())

    @property
    def param_billions(self) -> float:
        return self.total_params / 1_000_000_000


# ---------------------------------------------------------------------------
# Memory estimation contract (intelligence/ internal)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MemoryReport:
    """
    Memory estimate for a (model, strategy, batch_size) combination.
    Produced by estimator pipeline. Consumed by strategy selector.
    All values: int bytes (Law 10).
    """
    weights_bytes:       int
    activations_bytes:   int
    gradients_bytes:     int
    optimizer_bytes:     int
    overhead_bytes:      int
    method:              EstimationMethod
    confidence:          float        # 0.0 - 1.0
    error_bound_pct:     float        # expected ± error percentage

    def __post_init__(self) -> None:
        for fname in ("weights_bytes", "activations_bytes",
                      "gradients_bytes", "optimizer_bytes", "overhead_bytes"):
            val = getattr(self, fname)
            if isinstance(val, bool):
                raise TypeError(f"{fname} must be int, not bool")
            if val < 0:
                raise ValueError(f"{fname} must be >= 0, got {val}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")

    @property
    def total_bytes(self) -> int:
        """Peak VRAM needed. int bytes."""
        raw = (
            self.weights_bytes
            + self.activations_bytes
            + self.gradients_bytes
            + self.optimizer_bytes
            + self.overhead_bytes
        )
        if raw > sys.maxsize:
            raise OverflowError(
                f"MemoryReport total {raw} exceeds sys.maxsize {sys.maxsize}"
            )
        return raw

    @property
    def total_gib(self) -> float:
        return self.total_bytes / 1_073_741_824

    def fits_in(self, available_bytes: int, safety_threshold: float) -> bool:
        """True if this estimate fits within the safety threshold of available VRAM."""
        return self.total_bytes <= int(available_bytes * safety_threshold)

    def __add__(self, other: MemoryReport) -> MemoryReport:
        """Combine two reports (e.g. model + optimizer on separate shards)."""
        raw_total = self.total_bytes + other.total_bytes
        if raw_total > sys.maxsize:
            raise OverflowError(
                f"MemoryReport addition overflow: {raw_total} > {sys.maxsize}"
            )
        return MemoryReport(
            weights_bytes     = self.weights_bytes     + other.weights_bytes,
            activations_bytes = self.activations_bytes + other.activations_bytes,
            gradients_bytes   = self.gradients_bytes   + other.gradients_bytes,
            optimizer_bytes   = self.optimizer_bytes   + other.optimizer_bytes,
            overhead_bytes    = self.overhead_bytes    + other.overhead_bytes,
            method            = self.method,
            confidence        = min(self.confidence, other.confidence),
            error_bound_pct   = max(self.error_bound_pct, other.error_bound_pct),
        )


# ---------------------------------------------------------------------------
# Planning contracts (Layer 3 -> Layer 4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TensorPlacement:
    """Where a specific tensor lives and how to prefetch it."""
    tensor_name:      str
    tier:             MemoryTier
    size_bytes:       int
    transfer_ms:      float     # computed: size / bandwidth * 1000
    prefetch_lead_steps: int    # computed: ceil(transfer_ms / step_ms)

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise ValueError(f"size_bytes must be >= 0, got {self.size_bytes}")
        if self.transfer_ms < 0:
            raise ValueError(f"transfer_ms must be >= 0, got {self.transfer_ms}")
        if self.prefetch_lead_steps < 0:
            raise ValueError(
                f"prefetch_lead_steps must be >= 0, got {self.prefetch_lead_steps}"
            )


@dataclass(frozen=True, slots=True)
class PrefetchSchedule:
    """
    Complete prefetch schedule for one training step.
    CUDA stream IDs are assigned by stream_manager.py.
    """
    step:        int
    placements:  tuple[TensorPlacement, ...]
    stream_ids:  tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.placements) != len(self.stream_ids):
            raise ValueError(
                f"placements ({len(self.placements)}) and "
                f"stream_ids ({len(self.stream_ids)}) must have equal length"
            )


@dataclass(frozen=True, slots=True)
class AllocationPlan:
    """
    Complete tensor placement and prefetch plan.
    Produced by allocation_planner.py. Consumed by orchestration layer (L4).
    """
    placements:        tuple[TensorPlacement, ...]
    prefetch_schedule: tuple[PrefetchSchedule, ...]
    total_vram_bytes:  int
    total_ram_bytes:   int
    total_nvme_bytes:  int

    def __post_init__(self) -> None:
        for fname in ("total_vram_bytes", "total_ram_bytes", "total_nvme_bytes"):
            if getattr(self, fname) < 0:
                raise ValueError(f"{fname} must be >= 0")


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    """
    THE zero-config output. Produced by intelligence layer. Consumed by all of Layer 5.
    Carries every decision EGX made, with rationale.
    """
    # Strategy decision
    mode:                TrainingMode
    batch_size:          int
    grad_accum_steps:    int
    seq_len:             int
    optimizer:           OptimizerType
    scheduler:           SchedulerType
    parallel_strategy:   ParallelStrategy

    # LoRA config (None if mode == FULL_FINETUNE)
    lora_rank:           int | None
    lora_alpha:          int | None
    lora_dropout:        float | None
    lora_target_modules: tuple[str, ...] | None

    # Gradient checkpointing
    gradient_checkpointing: bool

    # Mixed precision
    use_bf16:            bool
    use_fp16:            bool

    # Estimation provenance
    confidence:          float
    estimation_method:   EstimationMethod
    decision_rationale:  str          # human-readable explanation

    # Memory plan (GiB for display; bytes used internally)
    vram_peak_bytes:     int
    ram_peak_bytes:      int
    nvme_peak_bytes:     int
    estimation_error_pct: float

    # Timing
    planned_at:          datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.grad_accum_steps < 1:
            raise ValueError(
                f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        if self.mode.uses_peft():
            if self.lora_rank is None:
                raise ValueError(f"lora_rank required for mode {self.mode}")
            if self.lora_target_modules is None:
                raise ValueError(f"lora_target_modules required for mode {self.mode}")

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps

    @property
    def vram_peak_gib(self) -> float:
        return self.vram_peak_bytes / 1_073_741_824


# ---------------------------------------------------------------------------
# Runtime snapshot (mutable — updated each monitoring cycle)
# ---------------------------------------------------------------------------

@dataclass(slots=True)   # NOT frozen — updated every poll interval
class HardwareSnapshot:
    """
    Real-time VRAM/RAM/utilization per GPU.
    The only mutable cross-layer object in EGX.
    Updated by orchestration/pressure/monitor.py every poll interval.
    """
    timestamp:        datetime
    vram_used:        dict[int, int]     # gpu_id -> bytes used
    vram_total:       dict[int, int]     # gpu_id -> bytes total
    ram_used:         int                # bytes
    ram_total:        int                # bytes
    gpu_utilization:  dict[int, float]   # gpu_id -> 0.0-1.0
    temp_celsius:     dict[int, float]   # gpu_id -> temperature

    def pressure_for_gpu(self, gpu_id: int) -> MemoryPressureLevel:
        used  = self.vram_used.get(gpu_id, 0)
        total = self.vram_total.get(gpu_id, 1)
        return MemoryPressureLevel.from_fraction(used, total)

    def worst_pressure(self) -> MemoryPressureLevel:
        """Returns the highest pressure level across all GPUs."""
        levels = [
            self.pressure_for_gpu(gid)
            for gid in self.vram_used
        ]
        if not levels:
            return MemoryPressureLevel.GREEN
        return max(levels, key=lambda l: ["green","yellow","orange","red","emergency"].index(l.value))




# ---------------------------------------------------------------------------
# Training result (Layer 5 -> Layer 7)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TrainingResult:
    """
    Complete training run result. Produced by elastic_loop.py.
    Returned to the user by EGX.train().
    """
    success:             bool
    final_loss:          float | None
    best_loss:           float | None
    total_steps:         int
    total_tokens:        int
    tokens_per_second:   float
    peak_vram_bytes:     int
    strategy_used:       TrainingMode
    optimizer_used:      OptimizerType
    batch_size_final:    int
    grad_accum_final:    int
    checkpoint_path:     Path | None
    export_path:         Path | None
    decision_rationale:  str
    recovery_events:     tuple[str, ...]   # log of any recovery actions taken
    elapsed_seconds:     float
    completed_at:        datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def peak_vram_gib(self) -> float:
        return self.peak_vram_bytes / 1_073_741_824

    def to_dict(self) -> dict[str, Any]:


        return {
            "success":           self.success,
            "final_loss":        self.final_loss,
            "best_loss":         self.best_loss,
            "total_steps":       self.total_steps,
            "total_tokens":      self.total_tokens,
            "tokens_per_second": self.tokens_per_second,
            "peak_vram_gib":     round(self.peak_vram_gib, 2),

            "strategy_used":     self.strategy_used.value,
            "optimizer_used":    self.optimizer_used.value,
            "batch_size_final":  self.batch_size_final,
            "grad_accum_final":  self.grad_accum_final,
            "checkpoint_path":   str(self.checkpoint_path) if self.checkpoint_path else None,
            "export_path":       str(self.export_path) if self.export_path else None,
            "decision_rationale": self.decision_rationale,
            "recovery_events":   list(self.recovery_events),
            "elapsed_seconds":   round(self.elapsed_seconds, 2),


            "completed_at":      self.completed_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Export result (Layer 5 export/ -> Layer 7)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ExportResult:
    """Result of a model export operation."""
    success:      bool
    output_path:  Path
    format:       str          # "safetensors" | "onnx"
    size_bytes:   int
    shards:       int          # 1 for single file, N for sharded
    sha256:       str          # checksum of output
    exported_at:  datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Recovery FSM context (resilience/ internal)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """
    Decision returned by RecoveryFSM for a given failure.
    Consumed by training/kernel.py.
    """
    action:           RecoveryAction
    new_state:        RecoveryState
    failure_type:     FailureType
    retry_count:      int
    is_terminal:      bool        # True = RecoveryAction.ABORT follows next
    log_message:      str
