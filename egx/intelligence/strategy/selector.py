"""
EGX Fibonacci Heap — DSA-1.

Why: O(1) amortized decrease-key and insert.
Used in: intelligence/strategy/selector.py
"""

from __future__ import annotations

import math
from typing import Any, List, Optional


class FibNode:
    __slots__ = ("key", "value", "parent", "child", "left", "right", "degree", "mark")
    
    def __init__(self, key: float, value: Any):
        self.key = key
        self.value = value
        self.parent: Optional[FibNode] = None
        self.child: Optional[FibNode] = None
        self.left: FibNode = self
        self.right: FibNode = self
        self.degree: int = 0
        self.mark: bool = False


class FibonacciHeap:
    """
    Law 11: Documented Big-O Complexity.
    Insert: O(1)
    Extract-Max: O(log n)
    Increase-Key: O(1) amortized
    Merge: O(1)
    """
    
    def __init__(self):
        self.max_node: Optional[FibNode] = None
        self.total_nodes: int = 0

    def insert(self, key: float, value: Any) -> FibNode:
        node = FibNode(key, value)
        if self.max_node is None:
            self.max_node = node
        else:
            self._link_to_root_list(node)
            if node.key > self.max_node.key:
                self.max_node = node
        self.total_nodes += 1
        return node

    def extract_max(self) -> Optional[FibNode]:
        z = self.max_node
        if z is not None:
            if z.child is not None:
                # Add all children to root list
                children = self._get_list(z.child)
                for child in children:
                    self._link_to_root_list(child)
                    child.parent = None
            
            # Remove z from root list
            z.left.right = z.right
            z.right.left = z.left
            
            if z == z.right:
                self.max_node = None
            else:
                self.max_node = z.right
                self._consolidate()
            self.total_nodes -= 1
        return z

    def increase_key(self, node: FibNode, new_key: float):
        if new_key < node.key:
            return # Should not happen in strategy selection
            
        node.key = new_key
        parent = node.parent
        if parent is not None and node.key > parent.key:
            self._cut(node, parent)
            self._cascading_cut(parent)
            
        if node.key > self.max_node.key:
            self.max_node = node

    def _link_to_root_list(self, node: FibNode):
        node.left = self.max_node
        node.right = self.max_node.right
        self.max_node.right = node
        node.right.left = node

    def _cut(self, x: FibNode, y: FibNode):
        # Remove x from child list of y
        if x.right == x:
            y.child = None
        else:
            x.left.right = x.right
            x.right.left = x.left
            if y.child == x:
                y.child = x.right
        y.degree -= 1
        self._link_to_root_list(x)
        x.parent = None
        x.mark = False

    def _cascading_cut(self, y: FibNode):
        z = y.parent
        if z is not None:
            if not y.mark:
                y.mark = True
            else:
                self._cut(y, z)
                self._cascading_cut(z)

    def _consolidate(self):
        # D(n) = log_phi(n)
        d = int(math.log(self.total_nodes, 1.618)) + 2
        a = [None] * d
        
        nodes = self._get_list(self.max_node)
        for w in nodes:
            x = w
            degree = x.degree
            while a[degree] is not None:
                y = a[degree]
                if x.key < y.key:
                    x, y = y, x
                self._fib_link(y, x)
                a[degree] = None
                degree += 1
            a[degree] = x
            
        self.max_node = None
        for i in range(d):
            if a[i] is not None:
                if self.max_node is None:
                    self.max_node = a[i]
                    a[i].left = a[i]
                    a[i].right = a[i]
                else:
                    self._link_to_root_list(a[i])
                    if a[i].key > self.max_node.key:
                        self.max_node = a[i]

    def _fib_link(self, y: FibNode, x: FibNode):
        # Remove y from root list
        y.left.right = y.right
        y.right.left = y.left
        
        # Make y child of x
        y.parent = x
        if x.child is None:
            x.child = y
            y.left = y
            y.right = y
        else:
            y.left = x.child
            y.right = x.child.right
            x.child.right = y
            y.right.left = y
        x.degree += 1
        y.mark = False

    def _get_list(self, start: FibNode) -> List[FibNode]:
        res = [start]
        curr = start.right
        while curr != start:
            res.append(curr)
            curr = curr.right
        return res
