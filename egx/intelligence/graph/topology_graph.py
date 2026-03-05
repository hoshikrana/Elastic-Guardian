"""
EGX Dijkstra Topology Optimizer — DSA-5.

Why: Optimal tensor movement routing in weighted hardware graphs.
Used in: intelligence/graph/topology_graph.py
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Tuple


class HardwareTopologyGraph:
    """
    Law 11: Dijkstra O((V+E) log V) routing.
    """

    def __init__(self):
        self.adj: Dict[str, List[Tuple[str, float]]] = {}

    def add_edge(self, src: str, dst: str, bw_gbps: float):
        # Weight is inverse of bandwidth (latency cost)
        weight = 1000.0 / max(0.1, bw_gbps)
        if src not in self.adj:
            self.adj[src] = []
        self.adj[src].append((dst, weight))

    def shortest_path(self, src: str, dst: str) -> Tuple[List[str], float]:
        """Dijkstra's shortest path."""
        distances = {src: 0.0}
        pq = [(0.0, src)]
        parent = {src: None}

        while pq:
            d, u = heapq.heappop(pq)
            if d > distances.get(u, float("inf")):
                continue
            if u == dst:
                break

            for v, weight in self.adj.get(u, []):
                new_dist = d + weight
                if new_dist < distances.get(v, float("inf")):
                    distances[v] = new_dist
                    parent[v] = u
                    heapq.heappush(pq, (new_dist, v))

        path = []
        curr = dst
        if dst in parent:
            while curr:
                path.append(curr)
                curr = parent[curr]
            path.reverse()

        return path, distances.get(dst, float("inf"))
