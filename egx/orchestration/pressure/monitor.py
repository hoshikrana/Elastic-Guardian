"""
EGX Skip List — DSA-3.

Why: Concurrent O(log n) time-ordered log.
Used in: orchestration/pressure/monitor.py
"""

from __future__ import annotations

import random
from typing import Any, List, Optional


class SkipNode:
    __slots__ = ("ts", "event", "forward")

    def __init__(self, ts: float, event: Any, levels: int):
        self.ts = ts
        self.event = event
        self.forward: List[Optional[SkipNode]] = [None] * levels


class PressureEventSkipList:
    """
    Law 11: Concurrent time-ordered pressure log.
    O(log n) insert / search. O(1) latest.
    """

    def __init__(self, max_levels: int = 16):
        self.max_levels = max_levels
        self.head = SkipNode(-float("inf"), None, max_levels)
        self.level = 0
        self._latest_node: Optional[SkipNode] = None

    def insert(self, ts: float, event: Any):
        update = [None] * self.max_levels
        curr = self.head

        for i in range(self.level, -1, -1):
            while curr.forward[i] and curr.forward[i].ts < ts:
                curr = curr.forward[i]
            update[i] = curr

        lvl = self._random_level()
        if lvl > self.level:
            for i in range(self.level + 1, lvl + 1):
                update[i] = self.head
            self.level = lvl

        new_node = SkipNode(ts, event, lvl + 1)
        for i in range(lvl + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

        # Update latest O(1) tracking
        if self._latest_node is None or ts > self._latest_node.ts:
            self._latest_node = new_node

    def latest(self) -> Optional[Any]:
        return self._latest_node.event if self._latest_node else None

    def find_at(self, ts: float) -> Optional[Any]:
        curr = self.head
        for i in range(self.level, -1, -1):
            while curr.forward[i] and curr.forward[i].ts < ts:
                curr = curr.forward[i]

        curr = curr.forward[0]
        if curr and curr.ts == ts:
            return curr.event
        return None

    def _random_level(self) -> int:
        lvl = 0
        while random.random() < 0.5 and lvl < self.max_levels - 1:
            lvl += 1
        return lvl
