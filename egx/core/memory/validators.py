"""
EGX Memory Validators — Layer 1.

Enforces Law 10: bool-first type validation to prevent type-coercion bugs.
"""

from __future__ import annotations

import sys
from typing import Any
from egx.core.exceptions import (
    MemoryValidationError,
    NegativeMemoryError,
    MemoryOverflowError,
    BoolAsIntError,
)


class StandardMemoryValidator:
    """
    Law 10 & Law 11 compliant validator.
    """

    @staticmethod
    def validate(value: Any, field_name: str) -> int:
        """
        Validates that a value is a safe memory integer.
        MUST check bool first because isinstance(True, int) is True.
        """
        if isinstance(value, bool):
            raise BoolAsIntError(field_name, value)

        if not isinstance(value, int):
            raise MemoryValidationError(
                f"Field '{field_name}' must be int, got {type(value)}"
            )

        if value < 0:
            raise NegativeMemoryError(field_name, value)

        if value > sys.maxsize:
            raise MemoryOverflowError(value, sys.maxsize)

        return value
