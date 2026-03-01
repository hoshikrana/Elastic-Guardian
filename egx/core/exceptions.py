"""
EGX typed exception hierarchy.

Design rules:
  - Every exception declares `recoverable: bool` at class level (never implied)
  - Every exception declares `failure_type: FailureType | None`
  - Every exception carries an ErrorContext for structured logging
  - No bare Exception raises anywhere in EGX — always raise a typed subclass
  - stdlib only (no torch imports)
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone

from egx.core.enums import FailureType, RecoveryAction

# ---------------------------------------------------------------------------
# Error context — carried by every exception for structured logging
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ErrorContext:
    """
    Machine-readable context attached to every EGXError.
    Consumed by RecoveryFSM and structured_logger.
    """
    component:  str                        # e.g. "analytical_estimator"
    timestamp:  datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata:   dict[str, Any] = field(default_factory=dict)


    stacktrace: str = field(default_factory=lambda: traceback.format_exc())

    def to_dict(self) -> dict[str, Any]:


        return {
            "component":  self.component,
            "timestamp":  self.timestamp.isoformat(),
            "metadata":   self.metadata,
        }


# ---------------------------------------------------------------------------
# Root exception
# ---------------------------------------------------------------------------

class EGXError(Exception):
    """
    Root of all EGX exceptions.

    Every subclass MUST declare:
      recoverable: bool   — whether RecoveryFSM should attempt recovery
      failure_type        — maps to RecoveryFSM dispatch table (None = non-training)
    """
    recoverable:    bool = False
    failure_type:   FailureType | None = None
    suggested_action: RecoveryAction = RecoveryAction.ABORT

    def __init__(
        self,
        message: str,
        context: ErrorContext | None = None,
        **metadata: Any,


    ) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or ErrorContext(
            component="unknown",
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:


        return {
            "error_type":        type(self).__name__,
            "message":           self.message,
            "recoverable":       self.recoverable,
            "suggested_action":  self.suggested_action.value,
            "failure_type":      self.failure_type.value if self.failure_type else None,
            "context":           self.context.to_dict(),
        }


# ---------------------------------------------------------------------------
# Hardware errors
# ---------------------------------------------------------------------------

class HardwareError(EGXError):
    """Base for all hardware-related failures."""


class GPUNotFoundError(HardwareError):
    """No CUDA GPU detected and CPU fallback is disabled."""
    recoverable     = False
    suggested_action = RecoveryAction.ABORT

    def __init__(self, probe_methods_tried: list[str]) -> None:
        super().__init__(
            f"No CUDA GPU found. Tried: {probe_methods_tried}",
            probe_methods_tried=probe_methods_tried,
        )
        self.probe_methods_tried = probe_methods_tried


class InsufficientVRAMError(HardwareError):
    """
    Not enough VRAM to run any supported training strategy.
    Carries exactly what hardware would be needed.
    """
    recoverable      = False
    suggested_action = RecoveryAction.ABORT

    def __init__(
        self,
        model_params:     int,
        available_vram:   int,
        minimum_required: int,
        recommendations:  list[str] | None = None,
    ) -> None:
        super().__init__(
            f"Insufficient VRAM: model needs ~{minimum_required // 1_073_741_824}GB, "
            f"available {available_vram // 1_073_741_824}GB",
            model_params=model_params,
            available_vram=available_vram,
            minimum_required=minimum_required,
        )
        self.model_params     = model_params
        self.available_vram   = available_vram
        self.minimum_required = minimum_required
        self.recommendations  = recommendations or []


class NVMLError(HardwareError):
    """pynvml operation failed. Often transient — driver reload can fix."""
    recoverable      = True
    failure_type     = FailureType.DRIVER_CRASH
    suggested_action = RecoveryAction.RETRY

    def __init__(self, operation: str, nvml_error: str) -> None:
        super().__init__(
            f"NVML error during '{operation}': {nvml_error}",
            operation=operation,
            nvml_error=nvml_error,
        )
        self.operation  = operation
        self.nvml_error = nvml_error


class ThermalThrottleError(HardwareError):
    """GPU temperature exceeds safe threshold."""
    recoverable      = True
    failure_type     = FailureType.THERMAL_THROTTLE
    suggested_action = RecoveryAction.RETRY

    def __init__(self, gpu_id: int, temp_c: float, threshold_c: float = 88.0) -> None:
        super().__init__(
            f"GPU {gpu_id} thermal throttle: {temp_c:.1f}°C (threshold {threshold_c}°C)",
            gpu_id=gpu_id,
            temp_c=temp_c,
            threshold_c=threshold_c,
        )
        self.gpu_id      = gpu_id
        self.temp_c      = temp_c
        self.threshold_c = threshold_c


class PCIeBandwidthError(HardwareError):
    """PCIe bandwidth too low for required prefetch lead time."""
    recoverable      = True
    suggested_action = RecoveryAction.RETRY

    def __init__(self, measured_gbps: float, required_gbps: float) -> None:
        super().__init__(
            f"PCIe bandwidth {measured_gbps:.1f} GB/s < required {required_gbps:.1f} GB/s",
            measured_gbps=measured_gbps,
            required_gbps=required_gbps,
        )
        self.measured_gbps  = measured_gbps
        self.required_gbps  = required_gbps


# ---------------------------------------------------------------------------
# Memory errors
# ---------------------------------------------------------------------------

class MemoryError(EGXError):
    """Base for memory arithmetic and validation failures."""


class MemoryValidationError(MemoryError):
    """A memory value failed validation."""
    recoverable = False


class NegativeMemoryError(MemoryValidationError):
    """Memory value is negative — physically impossible."""
    def __init__(self, field_name: str, value: int) -> None:
        super().__init__(
            f"Memory field '{field_name}' is negative: {value}",
            field_name=field_name,
            value=value,
        )
        self.field_name = field_name
        self.value      = value


class MemoryOverflowError(MemoryValidationError):
    """Memory value exceeds sys.maxsize — integer overflow guard."""
    def __init__(self, value: int, max_value: int) -> None:
        super().__init__(
            f"Memory value {value} exceeds sys.maxsize ({max_value})",
            value=value,
            max_value=max_value,
        )
        self.value     = value
        self.max_value = max_value


class BoolAsIntError(MemoryValidationError):
    """
    bool was passed where int is required.
    Python trap: isinstance(True, int) == True.
    The validator catches this BEFORE the int check.
    """
    def __init__(self, field_name: str, value: bool) -> None:
        super().__init__(
            f"Memory field '{field_name}' received bool ({value!r}). "
            "Pass an integer byte count, not a boolean.",
            field_name=field_name,
            value=value,
        )
        self.field_name = field_name
        self.value      = value


class OutOfMemoryError(MemoryError):
    """CUDA OOM during training. Triggers elastic batch reduction."""
    recoverable      = True
    failure_type     = FailureType.OOM
    suggested_action = RecoveryAction.HALVE_BATCH

    def __init__(
        self,
        batch_size:    int,
        vram_used:     int,
        vram_total:    int,
        step:          int | None = None,
    ) -> None:
        super().__init__(
            f"CUDA OOM at batch_size={batch_size} "
            f"({vram_used // 1_073_741_824}GB/{vram_total // 1_073_741_824}GB used)",
            batch_size=batch_size,
            vram_used=vram_used,
            vram_total=vram_total,
            step=step,
        )
        self.batch_size = batch_size
        self.vram_used  = vram_used
        self.vram_total = vram_total
        self.step       = step


# ---------------------------------------------------------------------------
# Planning errors
# ---------------------------------------------------------------------------

class PlanningError(EGXError):
    """Base for strategy selection and allocation planning failures."""


class InsufficientHardwareError(PlanningError):
    """
    No training strategy fits in available VRAM.
    Carries exact hardware requirements so user knows what to upgrade to.
    """
    recoverable      = False
    suggested_action = RecoveryAction.ABORT

    def __init__(
        self,
        model_params:   int,
        available_vram: int,
        recommendations: list[str],
    ) -> None:
        super().__init__(
            f"No strategy fits: model has {model_params / 1e9:.1f}B params, "
            f"available VRAM {available_vram // 1_073_741_824}GB. "
            f"Options: {recommendations}",
            model_params=model_params,
            available_vram=available_vram,
        )
        self.model_params    = model_params
        self.available_vram  = available_vram
        self.recommendations = recommendations


class EstimationError(PlanningError):
    """Base for memory estimation failures."""


class DryRunFailure(EstimationError):


    """Dry-run estimator threw an error. Falls back to analytical."""
    recoverable      = True
    suggested_action = RecoveryAction.RETRY

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Dry-run estimation failed: {reason}",
            reason=reason,
        )
        self.reason = reason


class DryRunTimeout(EstimationError):

    """Dry-run exceeded timeout. Falls back to analytical."""
    recoverable      = True
    suggested_action = RecoveryAction.RETRY

    def __init__(self, timeout_s: float) -> None:
        super().__init__(
            f"Dry-run timed out after {timeout_s}s",
            timeout_s=timeout_s,
        )
        self.timeout_s = timeout_s



class AllocationPlanError(PlanningError):
    """Could not build a valid AllocationPlan from the TrainingPlan."""
    recoverable = False


class PrefetchTimingError(PlanningError):
    """Required prefetch lead time exceeds available step time."""
    recoverable      = True
    suggested_action = RecoveryAction.RETRY

    def __init__(
        self,
        tensor_name:  str,
        required_ms:  float,
        available_ms: float,
    ) -> None:
        super().__init__(
            f"Tensor '{tensor_name}' needs {required_ms:.1f}ms prefetch, "
            f"only {available_ms:.1f}ms available",
            tensor_name=tensor_name,
            required_ms=required_ms,
            available_ms=available_ms,
        )
        self.tensor_name  = tensor_name
        self.required_ms  = required_ms
        self.available_ms = available_ms


# ---------------------------------------------------------------------------
# Training errors
# ---------------------------------------------------------------------------

class TrainingError(EGXError):
    """Base for runtime training failures."""


class NaNLossError(TrainingError):
    """Loss became NaN during training."""
    recoverable      = True
    failure_type     = FailureType.NAN_LOSS
    suggested_action = RecoveryAction.RELOAD_CHECKPOINT

    def __init__(self, step: int, loss_value: float) -> None:
        super().__init__(
            f"NaN loss at step {step} (value: {loss_value})",
            step=step,
            loss_value=loss_value,
        )
        self.step       = step
        self.loss_value = loss_value


class InfGradientError(TrainingError):
    """Gradient norm is infinite for a specific layer."""
    recoverable      = True
    failure_type     = FailureType.INF_GRADIENT
    suggested_action = RecoveryAction.RETRY

    def __init__(self, step: int, layer_name: str, grad_norm: float) -> None:
        super().__init__(
            f"Inf gradient at step {step}, layer '{layer_name}' "
            f"(norm: {grad_norm})",
            step=step,
            layer_name=layer_name,
            grad_norm=grad_norm,
        )
        self.step       = step
        self.layer_name = layer_name
        self.grad_norm  = grad_norm


class CheckpointCorruptError(TrainingError):
    """Checkpoint SHA256 checksum mismatch — file is corrupt."""
    recoverable      = True
    failure_type     = FailureType.CHECKPOINT_CORRUPT
    suggested_action = RecoveryAction.RELOAD_CHECKPOINT

    def __init__(
        self,
        path:            str,
        expected_sha256: str,
        actual_sha256:   str,
    ) -> None:
        super().__init__(
            f"Checkpoint corrupt: {path}. "
            f"Expected SHA256 {expected_sha256[:8]}..., got {actual_sha256[:8]}...",
            path=path,
            expected_sha256=expected_sha256,
            actual_sha256=actual_sha256,
        )
        self.path            = path
        self.expected_sha256 = expected_sha256
        self.actual_sha256   = actual_sha256


class DeadlockError(TrainingError):
    """Training kernel stopped sending heartbeats — deadlock detected."""
    recoverable      = True
    failure_type     = FailureType.DEADLOCK
    suggested_action = RecoveryAction.RESTART_PROCESS

    def __init__(self, timeout_s: float, last_heartbeat_step: int) -> None:
        super().__init__(
            f"Deadlock detected: no heartbeat for {timeout_s:.1f}s "
            f"(last step: {last_heartbeat_step})",
            timeout_s=timeout_s,
            last_heartbeat_step=last_heartbeat_step,
        )
        self.timeout_s            = timeout_s
        self.last_heartbeat_step  = last_heartbeat_step


# ---------------------------------------------------------------------------
# Plugin errors
# ---------------------------------------------------------------------------

class PluginError(EGXError):
    """Base for plugin registry failures."""
    recoverable = False


class PluginConflictError(PluginError):
    """Two plugins registered for the same hook/name."""
    def __init__(self, name: str, existing: str, new: str) -> None:
        super().__init__(
            f"Plugin conflict on '{name}': '{existing}' already registered, "
            f"cannot register '{new}'",
            name=name,
            existing=existing,
            new=new,
        )
        self.name     = name
        self.existing = existing
        self.new      = new


class CircularDependencyError(PluginError):
    """Module import graph contains a cycle. Detected by Kahn's algorithm."""
    def __init__(self, cycle: list[str]) -> None:
        cycle_str = " -> ".join(cycle)
        super().__init__(
            f"Circular import detected: {cycle_str}",
            cycle=cycle,
        )
        self.cycle = cycle


class PluginVersionError(PluginError):
    """Plugin version is incompatible with current EGX version."""
    def __init__(self, plugin: str, required: str, found: str) -> None:
        super().__init__(
            f"Plugin '{plugin}' requires EGX {required}, found {found}",
            plugin=plugin,
            required=required,
            found=found,
        )
        self.plugin   = plugin
        self.required = required
        self.found    = found


# ---------------------------------------------------------------------------
# Config errors
# ---------------------------------------------------------------------------

class ConfigError(EGXError):
    """Base for configuration validation failures."""
    recoverable = False


class SchemaValidationError(ConfigError):
    """YAML config value failed schema validation."""
    def __init__(self, key: str, value: Any, reason: str) -> None:


        super().__init__(
            f"Config validation failed for '{key}' = {value!r}: {reason}",
            key=key,
            value=value,
            reason=reason,
        )
        self.key    = key
        self.value  = value
        self.reason = reason


class IncompatibleConfigError(ConfigError):
    """Two config values are mutually incompatible."""
    def __init__(self, key_a: str, key_b: str, reason: str) -> None:
        super().__init__(
            f"Incompatible config: '{key_a}' and '{key_b}' cannot both be set. {reason}",
            key_a=key_a,
            key_b=key_b,
            reason=reason,
        )
        self.key_a  = key_a
        self.key_b  = key_b
        self.reason = reason
