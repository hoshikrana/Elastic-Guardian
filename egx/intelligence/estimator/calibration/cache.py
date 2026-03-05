"""
EGX Calibration Cache — Layer 3.

LRU cache for estimation results to avoid redundant dry-runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any, Dict, Optional

logger = logging.getLogger("egx.intelligence.calibration")


class CalibrationCache:
    """
    LRU cache for estimation results.
    Key = hash of (model_name, gpu_name, vram_bytes, mode).
    """

    def __init__(self, max_size: int = 256):
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def make_key(
        self,
        model_name: str,
        gpu_name: str,
        vram_bytes: int,
        mode: str,
    ) -> str:
        raw = json.dumps(
            {"model": model_name, "gpu": gpu_name, "vram": vram_bytes, "mode": mode},
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0
