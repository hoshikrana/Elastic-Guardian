"""
EGX Kahn's Algorithm — DSA-6.

Why: O(V+E) cycle detection at startup to prevent import-ordering bugs.
Used in: intelligence/graph/dependency_dag.py
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Set
from egx.core.exceptions import CircularDependencyError


class ModuleDependencyDAG:
    """
    Law 4 & 6: Kahn's BFS topo-validator.
    """

    def __init__(self):
        self.adj: Dict[str, Set[str]] = {}

    def add_dependency(self, module: str, depends_on: str):
        if module not in self.adj:
            self.adj[module] = set()
        self.adj[module].add(depends_on)

    def validate(self) -> List[str]:
        """
        Validates the DAG using Kahn's Algorithm.
        Raises CircularDependencyError if a cycle is found.
        """
        in_degree: Dict[str, int] = {node: 0 for node in self.adj}
        # Add nodes that are only dependencies
        all_nodes = set(self.adj.keys())
        for deps in self.adj.values():
            for d in deps:
                all_nodes.add(d)
                if d not in in_degree:
                    in_degree[d] = 0

        for u in self.adj:
            for v in self.adj[u]:
                in_degree[v] += 1

        queue = deque([u for u, d in in_degree.items() if d == 0])
        topo_order = []

        while queue:
            u = queue.popleft()
            topo_order.append(u)

            for v in self.adj.get(u, []):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(topo_order) != len(all_nodes):
            # Find a cycle for the exception
            cycle = self._find_cycle(all_nodes, in_degree)
            raise CircularDependencyError(cycle=cycle)

        return topo_order

    def _find_cycle(self, nodes: Set[str], in_degree: Dict[str, int]) -> List[str]:
        # Simple DFS to find the cycle in remaining nodes
        remaining = [u for u, d in in_degree.items() if d > 0]
        if not remaining:
            return []

        stack = [(remaining[0], [remaining[0]])]
        visited = set()

        while stack:
            u, path = stack.pop()
            if u in visited:
                continue
            visited.add(u)

            for v in self.adj.get(u, []):
                if v in path:
                    return path[path.index(v) :] + [v]
                stack.append((v, path + [v]))
        return []
