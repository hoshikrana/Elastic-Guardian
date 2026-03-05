"""
EGX Eviction Policy — Layer 4.

LRU eviction for VRAM tensor management.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import List

logger = logging.getLogger("egx.orchestration.pressure")


class LRUEvictionPolicy:
    """LRU-based eviction for VRAM tensor cache."""

    def __init__(self, capacity_bytes: int):
        self._capacity = capacity_bytes
        self._used = 0
        self._cache: OrderedDict[str, int] = OrderedDict()  # name -> size in bytes

    def access(self, tensor_name: str, size_bytes: int) -> None:
        """Record an access to a VRAM tensor."""
        if tensor_name in self._cache:
            self._cache.move_to_end(tensor_name)
        else:
            self._cache[tensor_name] = size_bytes
            self._used += size_bytes

    def evict_until(self, target_free: int) -> List[str]:
        """Evict LRU tensors until target_free bytes are available."""
        evicted = []
        while self._used + target_free > self._capacity and self._cache:
            name, size = self._cache.popitem(last=False)
            self._used -= size
            evicted.append(name)
            logger.info(f"Evicted: {name} ({size} bytes)")
        return evicted

    @property
    def usage_pct(self) -> float:
        return self._used / max(1, self._capacity)

    @property
    def free_bytes(self) -> int:
        return max(0, self._capacity - self._used)
