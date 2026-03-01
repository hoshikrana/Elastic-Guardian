"""
Memory formatting — pure display functions.

No state. No validation. No side effects.
Input: int bytes. Output: human-readable string.
These functions are for logging and UI output only.
Never use their output values in arithmetic.
"""

from __future__ import annotations

from egx.core.constants import GB, KB, MB


def format_bytes(b: int, precision: int = 2) -> str:
    """
    Format a byte count as a human-readable string.
    Automatically picks the most readable unit.

    Args:
        b:         Byte count (int).
        precision: Decimal places in the output.

    Returns:
        String like "14.00 GiB", "512.00 MiB", "1.50 KiB", "768 B"

    Examples:
        format_bytes(14_000_000_000)   -> "13.04 GiB"
        format_bytes(536_870_912)      -> "512.00 MiB"
        format_bytes(1_536)            -> "1.50 KiB"
        format_bytes(512)              -> "512 B"
    """
    if b >= GB:
        return f"{b / GB:.{precision}f} GiB"
    if b >= MB:
        return f"{b / MB:.{precision}f} MiB"
    if b >= KB:
        return f"{b / KB:.{precision}f} KiB"
    return f"{b} B"


def format_bytes_pair(used: int, total: int, precision: int = 2) -> str:
    """
    Format used/total as "X.XX GiB / Y.YY GiB (Z%)".
    Used in VRAM monitoring output.

    Examples:
        format_bytes_pair(7_516_192_768, 25_769_803_776)
        -> "7.00 GiB / 24.00 GiB (29%)"
    """
    pct = int(used / total * 100) if total > 0 else 0
    return f"{format_bytes(used, precision)} / {format_bytes(total, precision)} ({pct}%)"


def format_throughput(tokens_per_second: float) -> str:
    """
    Format training throughput.

    Examples:
        format_throughput(2847.3)   -> "2,847 tok/s"
        format_throughput(125000.0) -> "125,000 tok/s"
    """
    return f"{tokens_per_second:,.0f} tok/s"


def format_params(param_count: int) -> str:
    """
    Format parameter count in human-readable form.

    Examples:
        format_params(7_000_000_000)   -> "7.00B"
        format_params(125_000_000)     -> "125.00M"
        format_params(300_000)         -> "300.00K"
    """
    if param_count >= 1_000_000_000:
        return f"{param_count / 1_000_000_000:.2f}B"
    if param_count >= 1_000_000:
        return f"{param_count / 1_000_000:.2f}M"
    if param_count >= 1_000:
        return f"{param_count / 1_000:.2f}K"
    return str(param_count)


def format_memory_report(
    weights:     int,
    activations: int,
    gradients:   int,
    optimizer:   int,
    overhead:    int,
) -> str:
    """
    Format a full memory breakdown for logging.
    Returns a multi-line string.
    """
    total = weights + activations + gradients + optimizer + overhead
    lines = [
        "Memory breakdown:",
        f"  Weights:     {format_bytes(weights):>12}",
        f"  Activations: {format_bytes(activations):>12}",
        f"  Gradients:   {format_bytes(gradients):>12}",
        f"  Optimizer:   {format_bytes(optimizer):>12}",
        f"  Overhead:    {format_bytes(overhead):>12}",
        f"  {'─' * 28}",
        f"  TOTAL:       {format_bytes(total):>12}",
    ]
    return "\n".join(lines)
