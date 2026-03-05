"""
EGX Memory Formatting — Layer 1.

Human-readable display for memory values.
Display only—no logic or validation.
"""

from __future__ import annotations

from egx.core.memory.units import KB, MB, GB, TB


def format_bytes(bytes_val: int) -> str:
    """
    Law 10 violation check: Only ever use float for display.
    """
    if bytes_val >= TB:
        return f"{bytes_val / TB:.2f} TB"
    if bytes_val >= GB:
        return f"{bytes_val / GB:.2f} GB"
    if bytes_val >= MB:
        return f"{bytes_val / MB:.2f} MB"
    if bytes_val >= KB:
        return f"{bytes_val / KB:.1f} KB"
    return f"{bytes_val} B"
