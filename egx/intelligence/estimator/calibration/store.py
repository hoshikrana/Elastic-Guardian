"""
EGX Red-Black Tree — DSA-2.

Why: O(log n) for insert, delete, search, and range query.
Used in: intelligence/estimator/calibration/store.py
"""

from __future__ import annotations

import enum
from typing import Any, Optional, Tuple


class Color(enum.Enum):
    RED = 0
    BLACK = 1


class RBNode:
    __slots__ = ("key", "value", "color", "left", "right", "parent")
    
    def __init__(self, key: Any, value: Any, color: Color = Color.RED):
        self.key = key
        self.value = value
        self.color = color
        self.left: Optional[RBNode] = None
        self.right: Optional[RBNode] = None
        self.parent: Optional[RBNode] = None


class RedBlackTree:
    """
    Law 11: Calibration store with O(log n) range capability.
    """
    
    def __init__(self):
        self.nil = RBNode(None, None, Color.BLACK)
        self.root = self.nil

    def insert(self, key: Any, value: Any):
        node = RBNode(key, value)
        node.left = self.nil
        node.right = self.nil
        
        y = None
        x = self.root
        while x != self.nil:
            y = x
            if node.key < x.key:
                x = x.left
            else:
                x = x.right
        
        node.parent = y
        if y is None:
            self.root = node
        elif node.key < y.key:
            y.left = node
        else:
            y.right = node
            
        self._insert_fixup(node)

    def search(self, key: Any) -> Optional[Any]:
        curr = self.root
        while curr != self.nil:
            if key == curr.key: return curr.value
            if key < curr.key: curr = curr.left
            else: curr = curr.right
        return None

    def find_nearest(self, key: Any) -> Optional[Tuple[Any, Any]]:
        """Finds the node with key closest to the target."""
        best_node = None
        min_diff = float('inf')
        
        curr = self.root
        while curr != self.nil:
            diff = abs(curr.key - key)
            if diff < min_diff:
                min_diff = diff
                best_node = curr
            
            if key == curr.key: break
            if key < curr.key: curr = curr.left
            else: curr = curr.right
            
        return (best_node.key, best_node.value) if best_node else None

    def _insert_fixup(self, z: RBNode):
        while z.parent and z.parent.color == Color.RED:
            if z.parent == z.parent.parent.left:
                y = z.parent.parent.right
                if y.color == Color.RED:
                    z.parent.color = Color.BLACK
                    y.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    z = z.parent.parent
                else:
                    if z == z.parent.right:
                        z = z.parent
                        self._left_rotate(z)
                    z.parent.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    self._right_rotate(z.parent.parent)
            else:
                y = z.parent.parent.left
                if y.color == Color.RED:
                    z.parent.color = Color.BLACK
                    y.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    z = z.parent.parent
                else:
                    if z == z.parent.left:
                        z = z.parent
                        self._right_rotate(z)
                    z.parent.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    self._left_rotate(z.parent.parent)
        self.root.color = Color.BLACK

    def _left_rotate(self, x: RBNode):
        y = x.right
        x.right = y.left
        if y.left != self.nil:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _right_rotate(self, y: RBNode):
        x = y.left
        y.left = x.right
        if x.right != self.nil:
            x.right.parent = y
        x.parent = y.parent
        if y.parent is None:
            self.root = x
        elif y == y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        x.right = y
        y.parent = x
