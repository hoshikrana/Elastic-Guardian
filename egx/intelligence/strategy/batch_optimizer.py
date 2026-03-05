"""
EGX Binary Search Batch Optimizer — DSA-7.

Why: O(log n) batch scaling vs O(n) linear scan.
Used in: intelligence/strategy/batch_optimizer.py
"""

from __future__ import annotations

from typing import Callable


def find_max_batch_size(
    predicate: Callable[[int], bool],
    low: int = 1,
    high: int = 2048
) -> int:
    """
    Law 11: Binary Search on monotone predicate.
    """
    result = low
    while low <= high:
        mid = (low + high) >> 1
        if mid == 0: break
        
        if predicate(mid):
            result = mid
            low = mid + 1
        else:
            high = mid - 1
            
    return result
