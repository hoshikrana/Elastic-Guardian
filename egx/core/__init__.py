"""
egx.core — Layer 1. Foundation contracts.

Zero external dependencies. stdlib only.
No torch, no pynvml, no third-party imports of any kind.

This package defines:
  - All enumerations (enums.py)
  - All exceptions with recoverable: bool (exceptions.py)
  - All frozen dataclasses — the cross-layer contracts (models.py)
  - System-wide constants (constants.py)
  - Memory value object and utilities (memory/)
"""

from egx.core.enums import (
    ArchType,
    CheckpointStrategy,
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
from egx.core.exceptions import (
    AllocationPlanError,
    BoolAsIntError,
    CheckpointCorruptError,
    CircularDependencyError,
    ConfigError,
    DeadlockError,
    DryRunFailure,
    DryRunTimeout,
    EGXError,
    ErrorContext,
    GPUNotFoundError,
    HardwareError,
    IncompatibleConfigError,
    InfGradientError,
    InsufficientHardwareError,
    InsufficientVRAMError,
    MemoryError,
    MemoryOverflowError,
    MemoryValidationError,
    NaNLossError,
    NegativeMemoryError,
    NVMLError,
    OutOfMemoryError,
    PCIeBandwidthError,
    PlanningError,
    PluginConflictError,
    PluginError,
    PluginVersionError,
    PrefetchTimingError,
    SchemaValidationError,
    ThermalThrottleError,
    TrainingError,
)
from egx.core.memory import (
    MemoryValidator,
    MemoryValue,
    StandardMemoryValidator,
    activation_memory_bytes,
    format_bytes,
    optimizer_memory_bytes,
    params_to_bytes,
)
from egx.core.models import (
    AllocationPlan,
    ExportResult,
    GPUSpec,
    HardwareSnapshot,
    HardwareTopology,
    MemoryReport,
    ModelProfile,
    PrefetchSchedule,
    RecoveryDecision,
    TensorPlacement,
    TrainingPlan,
    TrainingResult,
)

__all__ = [
    # Enums
    "ArchType", "CheckpointStrategy", "DeviceType", "DType",
    "EstimationMethod", "FailureType", "HardwareTier", "InterconnectType",
    "MemoryPressureLevel", "MemoryTier", "OptimizerType", "ParallelStrategy",
    "RecoveryAction", "RecoveryState", "SchedulerType", "TrainingMode",
    # Exceptions
    "EGXError", "ErrorContext", "HardwareError", "GPUNotFoundError",
    "InsufficientVRAMError", "NVMLError", "ThermalThrottleError",
    "PCIeBandwidthError", "MemoryError", "MemoryValidationError",
    "NegativeMemoryError", "MemoryOverflowError", "BoolAsIntError",
    "OutOfMemoryError", "PlanningError", "InsufficientHardwareError",
    "EstimationError", "DryRunFailure", "DryRunTimeout",
    "AllocationPlanError", "PrefetchTimingError", "TrainingError",
    "NaNLossError", "InfGradientError", "CheckpointCorruptError",
    "DeadlockError", "PluginError", "PluginConflictError",
    "CircularDependencyError", "PluginVersionError", "ConfigError",
    "SchemaValidationError", "IncompatibleConfigError",
    # Models
    "GPUSpec", "HardwareTopology", "ModelProfile", "MemoryReport",
    "TensorPlacement", "PrefetchSchedule", "AllocationPlan", "TrainingPlan",
    "HardwareSnapshot", "TrainingResult", "ExportResult", "RecoveryDecision",
    # Memory
    "MemoryValue", "MemoryValidator", "StandardMemoryValidator",
    "format_bytes", "params_to_bytes", "optimizer_memory_bytes",
    "activation_memory_bytes",
]
