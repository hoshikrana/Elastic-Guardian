"""
EGX Input Sanitizer — Layer 4 (Resilience).

Validates and sanitizes training inputs before they enter the kernel.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
import torch

logger = logging.getLogger("egx.resilience.sanitizer")


class InputSanitizer:
    """
    Law 9: No silent fallbacks.
    Detects NaN/Inf in inputs BEFORE they corrupt gradients.
    """

    def __init__(self, strict: bool = True):
        self.strict = strict
        self._nan_count = 0
        self._inf_count = 0

    def check_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a training batch. Raises on corruption if strict."""
        clean = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor) and val.is_floating_point():
                has_nan = torch.isnan(val).any().item()
                has_inf = torch.isinf(val).any().item()

                if has_nan:
                    self._nan_count += 1
                    msg = f"NaN detected in batch key '{key}'"
                    if self.strict:
                        raise ValueError(msg)
                    logger.warning(f"{msg} — replacing with zeros.")
                    val = torch.nan_to_num(val, nan=0.0)

                if has_inf:
                    self._inf_count += 1
                    msg = f"Inf detected in batch key '{key}'"
                    if self.strict:
                        raise ValueError(msg)
                    logger.warning(f"{msg} — clamping.")
                    val = torch.clamp(val, min=-1e6, max=1e6)

            clean[key] = val
        return clean

    def check_loss(self, loss: torch.Tensor) -> bool:
        """Returns True if loss is valid (finite and non-NaN)."""
        if torch.isnan(loss).any():
            self._nan_count += 1
            return False
        if torch.isinf(loss).any():
            self._inf_count += 1
            return False
        return True

    @property
    def stats(self) -> Dict[str, int]:
        return {"nan_count": self._nan_count, "inf_count": self._inf_count}
