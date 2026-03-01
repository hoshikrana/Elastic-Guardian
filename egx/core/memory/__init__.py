"""
core.memory — Memory value object and utility functions.

Public surface:
    MemoryValue          — immutable memory quantity (int bytes)
    MemoryValidator      — validator ABC
    StandardMemoryValidator — production validator (bool trap first)
    NullMemoryValidator  — no-op for testing
    format_bytes         — display formatting
    params_to_bytes      — parameter count -> bytes
    optimizer_memory_bytes — optimizer state bytes
    activation_memory_bytes — activation memory estimation
"""

from egx.core.memory.formatting import (
    format_bytes,
    format_bytes_pair,
    format_memory_report,
    format_params,
    format_throughput,
)
from egx.core.memory.units import (
    activation_memory_bytes,
    bytes_to_gib,
    bytes_to_mib,
    gib_to_bytes,
    mib_to_bytes,
    optimizer_memory_bytes,
    params_to_bytes,
)
from egx.core.memory.validators import (
    MemoryValidator,
    NullMemoryValidator,
    StandardMemoryValidator,
    get_default_validator,
)
from egx.core.memory.value import MemoryValue

__all__ = [
    "MemoryValue",
    "MemoryValidator",
    "StandardMemoryValidator",
    "NullMemoryValidator",
    "get_default_validator",
    "format_bytes",
    "format_bytes_pair",
    "format_memory_report",
    "format_params",
    "format_throughput",
    "params_to_bytes",
    "optimizer_memory_bytes",
    "activation_memory_bytes",
    "bytes_to_gib",
    "bytes_to_mib",
    "gib_to_bytes",
    "mib_to_bytes",
]
