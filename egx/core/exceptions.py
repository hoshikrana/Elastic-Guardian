"""
EGX Core Exceptions — Layer 1.

Every exception has:
1. recoverable: bool (explicit)
2. suggested_action: RecoveryAction (enum)
3. context: ErrorContext (dataclass)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from egx.core.enums import RecoveryAction


@dataclass(frozen=True)
class ErrorContext:
    timestamp: float = field(default_factory=time.time)
    component: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class EGXError(Exception):
    """Base for all EGX errors. Law 8: recoverability must be explicit."""
    
    def __init__(
        self, 
        message: str, 
        recoverable: bool, 
        suggested_action: RecoveryAction = RecoveryAction.ABORT,
        context: Optional[ErrorContext] = None
    ):
        super().__init__(message)
        self.message = message
        self.recoverable = recoverable
        self.suggested_action = suggested_action
        self.context = context or ErrorContext()


# --- Hardware Errors (Layer 2 precursors) ---

class HardwareError(EGXError):
    """Root for hardware-related failures."""
    pass

class GPUNotFoundError(HardwareError):
    def __init__(self, message: str = "No CUDA/MPS GPUs detected", context: Optional[ErrorContext] = None):
        super().__init__(message, recoverable=False, suggested_action=RecoveryAction.ABORT, context=context)

class InsufficientVRAMError(HardwareError):
    def __init__(self, required_bytes: int, available_bytes: int, context: Optional[ErrorContext] = None):
        msg = f"Insufficient VRAM: Need {required_bytes}, have {available_bytes}"
        super().__init__(msg, recoverable=False, suggested_action=RecoveryAction.ABORT, context=context)

class NVMLError(HardwareError):
    def __init__(self, inner: str, context: Optional[ErrorContext] = None):
        super().__init__(f"NVML Transient Error: {inner}", recoverable=True, suggested_action=RecoveryAction.RETRY, context=context)

class ThermalThrottleError(HardwareError):
    def __init__(self, gpu_id: int, temp_c: float, context: Optional[ErrorContext] = None):
        super().__init__(f"GPU {gpu_id} Throttling: {temp_c}°C", recoverable=True, suggested_action=RecoveryAction.ABORT, context=context)


# --- Memory Errors (Layer 1 Logic) ---

class EGXMemoryError(EGXError):
    """Root for memory arithmetic and logic errors."""
    pass

class MemoryValidationError(EGXMemoryError):
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(message, recoverable=False, suggested_action=RecoveryAction.ABORT, context=context)

class NegativeMemoryError(MemoryValidationError):
    def __init__(self, field: str, value: int):
        super().__init__(f"Memory value for '{field}' cannot be negative: {value}")

class MemoryOverflowError(MemoryValidationError):
    def __init__(self, val: int, limit: int):
        super().__init__(f"Memory value {val} exceeds system limit {limit}")

class BoolAsIntError(MemoryValidationError):
    def __init__(self, field: str, value: bool):
        super().__init__(f"Type Trap: Field '{field}' received bool ({value}) instead of int. Law 10 violation.")

class OutOfMemoryError(EGXMemoryError):
    def __init__(self, msg: str = "VRAM Out of Memory", context: Optional[ErrorContext] = None):
        super().__init__(msg, recoverable=True, suggested_action=RecoveryAction.HALVE_BATCH, context=context)


# --- Planning & Intelligence Errors (Layer 3) ---

class PlanningError(EGXError):
    pass

class InsufficientHardwareError(PlanningError):
    def __init__(self, recs: list[str], context: Optional[ErrorContext] = None):
        super().__init__("Hardware insufficient for model. Recommendations: " + ", ".join(recs), 
                         recoverable=False, suggested_action=RecoveryAction.ABORT, context=context)

class EstimationError(PlanningError):
    pass

class DryRunFailure(EstimationError):
    def __init__(self, inner: str, context: Optional[ErrorContext] = None):
        super().__init__(f"Dry-run failed: {inner}", recoverable=True, suggested_action=RecoveryAction.RETRY, context=context)


# --- Training & Resilience Errors (Layer 4-5) ---

class TrainingError(EGXError):
    pass

class NaNLossError(TrainingError):
    def __init__(self, step: int, context: Optional[ErrorContext] = None):
        super().__init__(f"NaN Loss detected at step {step}", recoverable=True, suggested_action=RecoveryAction.RELOAD_CHECKPOINT, context=context)

class InfGradientError(TrainingError):
    def __init__(self, step: int, layer: str, context: Optional[ErrorContext] = None):
        super().__init__(f"Inf Gradient in layer '{layer}' at step {step}", recoverable=True, suggested_action=RecoveryAction.RETRY, context=context)

class CheckpointCorruptError(TrainingError):
    def __init__(self, path: str, context: Optional[ErrorContext] = None):
        super().__init__(f"Checkpoint corrupted (SHA256 mismatch): {path}", recoverable=True, suggested_action=RecoveryAction.RELOAD_CHECKPOINT, context=context)

class DeadlockError(TrainingError):
    def __init__(self, timeout_s: float, last_step: int, context: Optional[ErrorContext] = None):
        super().__init__(f"Training Deadlock (Heartbeat > {timeout_s}s). Last Step: {last_step}", 
                         recoverable=True, suggested_action=RecoveryAction.RESTART_PROCESS, context=context)


# --- Plugin & Config Errors ---

class PluginError(EGXError):
    pass

class CircularDependencyError(PluginError):
    def __init__(self, cycle: list[str], context: Optional[ErrorContext] = None):
        super().__init__(f"Circular Dependency in modules: {' -> '.join(cycle)}", 
                         recoverable=False, suggested_action=RecoveryAction.ABORT, context=context)

class ConfigError(EGXError):
    pass

class SchemaValidationError(ConfigError):
    def __init__(self, details: str, context: Optional[ErrorContext] = None):
        super().__init__(f"Config Schema Error: {details}", recoverable=False, suggested_action=RecoveryAction.ABORT, context=context)
