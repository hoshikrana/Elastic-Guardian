"""
EGX Memory Units — Layer 1.

Pure functions for unit conversion.
Results are always int bytes (Law 10).
"""

from __future__ import annotations

from typing import Union
from egx.core.memory.validators import StandardMemoryValidator


KB = 1024
MB = 1024 * 1024
GB = 1024 * 1024 * 1024
TB = 1024 * 1024 * 1024 * 1024


def to_bytes(value: Union[int, float], unit: int = 1) -> int:
    """
    Converts a value to absolute bytes.
    Implicitly validates Law 10 compliance.
    """
    result = int(value * unit)
    return StandardMemoryValidator.validate(result, "unit_conversion")


def from_bytes(bytes_val: int, unit: int = 1) -> float:
    """
    Converts bytes to a specific unit for display.
    Only returns float for display layer usage.
    """
    return bytes_val / unit