"""
MemoryValue — immutable memory quantity.

Always stores int bytes. All arithmetic preserves int bytes.
Overflow guard on __add__. Injected validator (Law 3).
"""

from __future__ import annotations

import sys

from egx.core.exceptions import MemoryOverflowError
from egx.core.memory.validators import MemoryValidator, get_default_validator


class MemoryValue:
    """
    Immutable wrapper around a memory byte count.

    Why a class and not just int?
    - Makes intent explicit at type level (can't pass raw int where MemoryValue expected)
    - Enforces validation on construction
    - Provides safe arithmetic with overflow detection
    - Carries display methods without polluting int

    All internal storage is int bytes.
    """

    __slots__ = ("_bytes", "_validator")

    def __init__(
        self,
        value: int,
        validator: MemoryValidator | None = None,
        field_name: str = "MemoryValue",
    ) -> None:
        v = validator or get_default_validator()
        self._bytes:    int             = v.validate(value, field_name)
        self._validator: MemoryValidator = v

    # ── Core access ──────────────────────────────────────────────────────

    @property
    def bytes(self) -> int:
        """Raw byte count. Always int."""
        return self._bytes

    @property
    def gib(self) -> float:
        """Display value in GiB. For output only."""
        return self._bytes / 1_073_741_824

    @property
    def mib(self) -> float:
        """Display value in MiB. For output only."""
        return self._bytes / 1_048_576

    # ── Arithmetic ────────────────────────────────────────────────────────

    def __add__(self, other: MemoryValue | int) -> MemoryValue:
        other_bytes = other._bytes if isinstance(other, MemoryValue) else other
        result = self._bytes + other_bytes
        if result > sys.maxsize:
            raise MemoryOverflowError(result, sys.maxsize)
        return MemoryValue(result, self._validator)

    def __sub__(self, other: MemoryValue | int) -> MemoryValue:
        other_bytes = other._bytes if isinstance(other, MemoryValue) else other
        result = self._bytes - other_bytes
        if result < 0:
            # MemoryValue cannot be negative — represents physical storage
            from egx.core.exceptions import NegativeMemoryError
            raise NegativeMemoryError("subtraction_result", result)
        return MemoryValue(result, self._validator)

    def __mul__(self, factor: int | float) -> MemoryValue:
        import math
        result = math.ceil(self._bytes * factor)
        if result > sys.maxsize:
            raise MemoryOverflowError(result, sys.maxsize)
        return MemoryValue(result, self._validator)

    def __floordiv__(self, divisor: int) -> MemoryValue:
        if divisor <= 0:
            raise ValueError(f"Divisor must be > 0, got {divisor}")
        return MemoryValue(self._bytes // divisor, self._validator)

    # ── Comparison ────────────────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MemoryValue):
            return self._bytes == other._bytes
        if isinstance(other, int):
            return self._bytes == other
        return NotImplemented

    def __lt__(self, other: MemoryValue | int) -> bool:
        other_bytes = other._bytes if isinstance(other, MemoryValue) else other
        return self._bytes < other_bytes

    def __le__(self, other: MemoryValue | int) -> bool:
        other_bytes = other._bytes if isinstance(other, MemoryValue) else other
        return self._bytes <= other_bytes

    def __gt__(self, other: MemoryValue | int) -> bool:
        other_bytes = other._bytes if isinstance(other, MemoryValue) else other
        return self._bytes > other_bytes

    def __ge__(self, other: MemoryValue | int) -> bool:
        other_bytes = other._bytes if isinstance(other, MemoryValue) else other
        return self._bytes >= other_bytes

    # ── Identity ──────────────────────────────────────────────────────────

    def __hash__(self) -> int:
        return hash(self._bytes)

    def __repr__(self) -> str:
        return f"MemoryValue({self._bytes} bytes / {self.gib:.3f} GiB)"

    def __int__(self) -> int:
        return self._bytes

    def __index__(self) -> int:
        """Allows use in slice and array index contexts."""
        return self._bytes

    # ── Constructors ─────────────────────────────────────────────────────

    @classmethod
    def from_gib(cls, gib: float) -> MemoryValue:
        """Construct from GiB (float). Rounds up."""
        import math
        return cls(math.ceil(gib * 1_073_741_824))

    @classmethod
    def from_mib(cls, mib: float) -> MemoryValue:
        """Construct from MiB (float). Rounds up."""
        import math
        return cls(math.ceil(mib * 1_048_576))

    @classmethod
    def zero(cls) -> MemoryValue:
        return cls(0)
