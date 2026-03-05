"""
EGX Config Trie — DSA-8.

Why: O(k) namespace resolution and O(k+results) prefix queries for CLI.
Used in: runtime/config_loader.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class TrieNode:
    __slots__ = ("children", "value", "is_end")
    
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.value: Any = None
        self.is_end: bool = False


class ConfigTrie:
    """
    Law 11: Trie-backed config namespace resolver.
    """
    
    def __init__(self):
        self.root = TrieNode()

    def insert(self, dotted_key: str, value: Any):
        """O(k) where k is number of segments."""
        node = self.root
        segments = dotted_key.split(".")
        for seg in segments:
            if seg not in node.children:
                node.children[seg] = TrieNode()
            node = node.children[seg]
        node.value = value
        node.is_end = True

    def get(self, dotted_key: str) -> Optional[Any]:
        """O(k) where k is number of segments."""
        node = self.root
        segments = dotted_key.split(".")
        for seg in segments:
            if seg not in node.children:
                return None
            node = node.children[seg]
        return node.value if node.is_end else None

    def find_by_prefix(self, prefix: str) -> List[str]:
        """O(k + results) prefix search."""
        node = self.root
        segments = prefix.split(".") if prefix else []
        for seg in segments:
            if seg not in node.children:
                return []
            node = node.children[seg]
            
        results = []
        self._dfs(node, prefix, results)
        return results

    def _dfs(self, node: TrieNode, current_key: str, results: List[str]):
        if node.is_end:
            results.append(current_key)
            
        for seg, child in node.children.items():
            new_key = f"{current_key}.{seg}" if current_key else seg
            self._dfs(child, new_key, results)
