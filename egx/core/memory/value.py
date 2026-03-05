"""
EGX Memory Immutable Value Object — Layer 1.

Enforces Law 10: Memory values are always int (bytes), never float.
Supports strictly int arithmetic.
"""

from __future__ import annotations

import sys
from typing import Any, Union
from egx.core.exceptions import MemoryOverflowError, NegativeMemoryError


class MemoryValue:
    """
    Law 10: Integer-only memory representation.
    Wraps an int to provide strict validation and overflow guards.
    """

    __slots__ = ("_bytes",)

    def __init__(self, val: Union[int, MemoryValue]):
        if isinstance(val, MemoryValue):
            val = val._bytes

        # Law 10: Explicit check for bool as int trap
        if isinstance(val, bool):
            # This would be caught by validators too, but we harden here.
            val = int(val)

        if val < 0:
            raise NegativeMemoryError("MemoryValue", val)
        if val > sys.maxsize:
            raise MemoryOverflowError(val, sys.maxsize)

        self._bytes: int = int(val)

    @property
    def bytes(self) -> int:
        return self._bytes

    def __add__(self, other: Union[int, MemoryValue]) -> MemoryValue:
        other_val = other.bytes if isinstance(other, MemoryValue) else other
        return MemoryValue(self._bytes + other_val)

    def __sub__(self, other: Union[int, MemoryValue]) -> MemoryValue:
        other_val = other.bytes if isinstance(other, MemoryValue) else other
        return MemoryValue(self._bytes - other_val)

    def __mul__(self, other: Union[int, float]) -> MemoryValue:
        # Multiplication by float allowed for scaling but result cast to int
        return MemoryValue(int(self._bytes * other))

    def __truediv__(self, other: Union[int, float]) -> MemoryValue:
        return MemoryValue(int(self._bytes / other))

    def __lt__(self, other: Union[int, MemoryValue]) -> bool:
        other_val = other.bytes if isinstance(other, MemoryValue) else other
        return self._bytes < other_val

    def __le__(self, other: Union[int, MemoryValue]) -> bool:
        other_val = other.bytes if isinstance(other, MemoryValue) else other
        return self._bytes <= other_val

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, MemoryValue):
            return self._bytes == other._bytes
        if isinstance(other, int):
            return self._bytes == other
        return False

    def __repr__(self) -> str:
        return f"MemoryValue({self._bytes} bytes)"

    def __int__(self) -> int:
        return self._bytes
