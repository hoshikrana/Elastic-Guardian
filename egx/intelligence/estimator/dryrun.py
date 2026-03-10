"""
EGX Segment Tree — DSA-4.

Why: O(log n) range-max for windowed peak memory detection.
Used in: intelligence/estimator/dryrun.py
"""

from __future__ import annotations


class MemorySegmentTree:
    """
    Law 11: Range-max detector for estimation snapshots.
    """

    def __init__(self, size: int):
        self.n = 1
        while self.n < size:
            self.n <<= 1
        self.tree = [0] * (2 * self.n)

    def update(self, pos: int, val_bytes: int):
        """O(log n) point update."""
        i = pos + self.n
        self.tree[i] = val_bytes
        while i > 1:
            i >>= 1
            self.tree[i] = max(self.tree[2 * i], self.tree[2 * i + 1])

    def query_max(self, left: int, right: int) -> int:
        """O(log n) range-max query [left, right)."""
        res = 0
        left += self.n
        right += self.n
        while left < right:
            if left & 1:
                res = max(res, self.tree[left])
                left += 1
            if right & 1:
                right -= 1
                res = max(res, self.tree[right])
            left >>= 1
            right >>= 1
        return res

    def global_peak(self) -> int:
        """O(1) global max."""
        return self.tree[1]
