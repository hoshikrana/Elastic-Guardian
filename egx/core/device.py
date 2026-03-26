"""
EGX Device Utilities — Layer 1.

Centralized device detection to avoid duplicated logic across the codebase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_DEVICE_CACHE: str | None = None


def get_default_device() -> str:
    """Return the best available device string.

    Resolution order: CUDA → MPS → CPU.
    The result is cached after the first call.
    """
    global _DEVICE_CACHE  # noqa: PLW0603
    if _DEVICE_CACHE is not None:
        return _DEVICE_CACHE

    try:
        import torch
    except ImportError:
        _DEVICE_CACHE = "cpu"
        return _DEVICE_CACHE

    if torch.cuda.is_available():
        _DEVICE_CACHE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _DEVICE_CACHE = "mps"
    else:
        _DEVICE_CACHE = "cpu"

    return _DEVICE_CACHE


def get_device_type() -> str:
    """Return the device *type* string for ``torch.amp.autocast``."""
    dev = get_default_device()
    return "cuda" if dev == "cuda" else "cpu"
