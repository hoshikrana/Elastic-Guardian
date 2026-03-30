"""
EGX Core Models — Layer 1.

Every public contract is a frozen dataclass with slots=True.
Law 5: Immutable contracts were mandatory.
Law 10: All memory fields are int bytes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .enums import (
    HardwareTier,
    InterconnectType,
    MemoryTier,
    DType,
    TrainingMode,
    ParallelStrategy,
    OptimizerType,
    EstimationMethod,
    ArchType,
)


@dataclass(frozen=True, slots=True)
class GPUSpec:
    """Snapshot of a single GPU's capabilities and current state."""

    device_id: int
    name: str
    vram_bytes: int  # bytes ONLY (Law 10)
    compute_capability: Tuple[int, int]
    memory_bandwidth_gbps: float
    fp16_tflops: float
    bf16_tflops: float
    supports_flash_attn2: bool
    supports_fp8: bool
    nvlink_peer_ids: Tuple[int, ...]
    vendor: str = "nvidia"  # nvidia, apple, intel

    @property
    def vram_gb(self) -> float:
        return self.vram_bytes / (1024 ** 3)

    @property
    def tier(self) -> HardwareTier:
        gb = self.vram_gb
        if gb <= 12:
            return HardwareTier.LAPTOP
        if gb <= 48:
            return HardwareTier.WORKSTATION
        if gb < 80:
            return HardwareTier.PROSUMER
        return HardwareTier.DATACENTER


@dataclass(frozen=True, slots=True)
class HardwareTopology:
    gpus: Tuple[GPUSpec, ...]
    cpu_cores: int
    ram_bytes: int
    nvme_bytes: int
    nvme_seq_read_gbps: float
    nvme_seq_write_gbps: float
    pcie_bandwidth_gbps: float
    gpu_interconnect_gbps: float
    interconnect: InterconnectType
    node_count: int = 1

    @property
    def total_vram_bytes(self) -> int:
        return sum(g.vram_bytes for g in self.gpus)

    @property
    def has_nvlink(self) -> bool:
        return self.interconnect == InterconnectType.NVLINK

    @property
    def hardware_tier(self) -> HardwareTier:
        if not self.gpus:
            return HardwareTier.LAPTOP
        # Returns highest tier in topology
        return max(g.tier for g in self.gpus)


@dataclass(frozen=True, slots=True)
class ModelProfile:
    arch: ArchType
    params: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    max_seq_len: int
    dtype: DType


@dataclass(frozen=True, slots=True)
class MemoryReport:
    """Aggregated memory state of a hardware node or cluster."""

    weights_bytes: int
    activations_bytes: int
    optimizer_bytes: int
    gradients_bytes: int
    overhead_bytes: int
    total_bytes: int
    method: EstimationMethod
    confidence: float
    error_bound_pct: float
    correction_factor: float = 1.0

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024**3)

    def __add__(self, other: MemoryReport) -> MemoryReport:
        from .exceptions import MemoryOverflowError

        total = self.total_bytes + other.total_bytes
        if total > sys.maxsize:
            raise MemoryOverflowError(total, sys.maxsize)

        return MemoryReport(
            weights_bytes=self.weights_bytes + other.weights_bytes,
            activations_bytes=self.activations_bytes + other.activations_bytes,
            optimizer_bytes=self.optimizer_bytes + other.optimizer_bytes,
            gradients_bytes=self.gradients_bytes + other.gradients_bytes,
            overhead_bytes=self.overhead_bytes + other.overhead_bytes,
            total_bytes=total,
            method=self.method,
            confidence=min(self.confidence, other.confidence),
            error_bound_pct=max(self.error_bound_pct, other.error_bound_pct),
        )


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    mode: TrainingMode
    dtype: DType
    parallel_strategy: ParallelStrategy
    batch_size: int
    grad_accum_steps: int
    seq_len: int
    gradient_checkpointing: bool
    mixed_precision: bool
    flash_attention: bool
    cpu_offload_optimizer: bool
    lora_rank: Optional[int] = None
    lora_alpha: Optional[float] = None
    lora_targets: Tuple[str, ...] = field(default_factory=tuple)
    quantization_bits: Optional[int] = None
    optimizer: OptimizerType = OptimizerType.ADAMW
    max_grad_norm: float = 1.0
    confidence: float = 1.0
    estimation_method: EstimationMethod = EstimationMethod.ANALYTICAL
    decision_rationale: str = ""

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps

    @property
    def uses_peft(self) -> bool:
        return self.lora_rank is not None


@dataclass(frozen=True, slots=True)
class TensorPlacement:
    tensor_name: str
    tier: MemoryTier
    size_bytes: int
    source_device_id: int
    transfer_bw_gbps: float
    transfer_ms: float
    prefetch_lead_steps: int
    evictable: bool = True


@dataclass(frozen=True, slots=True)
class PrefetchSchedule:
    event: str
    tensors: Tuple[str, ...]
    lead_time_ms: float
    cuda_stream_id: int


@dataclass(frozen=True, slots=True)
class AllocationPlan:
    placements: Tuple[TensorPlacement, ...]
    prefetch_schedules: Tuple[PrefetchSchedule, ...]
    estimated_vram_peak_bytes: int
    estimated_ram_peak_bytes: int
    estimated_nvme_peak_bytes: int
    plan_confidence: float
    fallback_plan: Optional[AllocationPlan] = None


@dataclass(frozen=True)
class TrainingResult:
    success: bool
    steps_completed: int
    final_loss: float
    best_loss: float
    time_s: float
    tokens_per_second: float
    oom_count: int
    nan_count: int
    peak_vram_gb: float
    strategy_used: TrainingMode
    decision_rationale: str = ""
