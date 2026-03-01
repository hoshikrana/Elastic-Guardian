"""
Memory value validators.

CRITICAL ORDER: bool check MUST come before int check.
Python trap: isinstance(True, int) == True
If you check int first, booleans silently pass through.

Design: validators are injected, never imported globally (Law 3).
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod

from egx.core.exceptions import (
    BoolAsIntError,
    MemoryOverflowError,
    NegativeMemoryError,
)


class MemoryValidator(ABC):
    """
    Abstract base for memory value validators.
    Injected into MemoryValue and unit converters — never instantiated globally.
    """

    @abstractmethod
    def validate(self, value: object, field_name: str = "value") -> int:
        """
        Validate a memory value.

        Args:
            value:      The value to validate.
            field_name: Name shown in error messages.

        Returns:
            The validated value as int.

        Raises:
            BoolAsIntError:      value is a bool (checked before int).
            MemoryValidationError: value is negative or overflows.
            TypeError:           value is not an int at all.
        """
        ...


class StandardMemoryValidator(MemoryValidator):
    """
    Production validator. Enforces three rules in this exact order:
      1. bool check  — isinstance(True, int) == True, so bool must be caught first
      2. type check  — must be int
      3. range check — must be >= 0 and <= sys.maxsize
    """

    def validate(self, value: object, field_name: str = "value") -> int:
        # ── Rule 1: bool BEFORE int ──────────────────────────────────────
        # Python's bool is a subclass of int. isinstance(True, int) is True.
        # Without this guard, True would pass the int check and become 1.
        if isinstance(value, bool):
            raise BoolAsIntError(field_name, value)  # type: ignore[arg-type]

        # ── Rule 2: must be int ──────────────────────────────────────────

        if not isinstance(value, int):
            raise TypeError(
                f"Memory field '{field_name}' must be int, "
                f"got {type(value).__name__}: {value!r}"
            )

        # ── Rule 3: range ────────────────────────────────────────────────
        if value < 0:
            raise NegativeMemoryError(field_name, value)

        if value > sys.maxsize:
            raise MemoryOverflowError(value, sys.maxsize)

        return value


class NullMemoryValidator(MemoryValidator):
    """
    No-op validator for testing or trusted internal paths.
    Still catches bools (that trap is always enforced).
    """

    def validate(self, value: object, field_name: str = "value") -> int:
        # Still catch the bool trap even in null validator
        if isinstance(value, bool):
            raise BoolAsIntError(field_name, value)  # type: ignore[arg-type]
        if not isinstance(value, int):

            raise TypeError(
                f"Memory field '{field_name}' must be int, "
                f"got {type(value).__name__}"
            )
        return value


# Module-level singleton for use in unit converters
# Injected explicitly — not imported from other modules as a global
_DEFAULT_VALIDATOR: MemoryValidator = StandardMemoryValidator()


def get_default_validator() -> MemoryValidator:
    """Return the module-level default validator. Used by units.py."""
    return _DEFAULT_VALIDATOR
