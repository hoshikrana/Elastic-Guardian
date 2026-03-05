"""
EGX Test: intelligence/graph/topology_graph.py (DSA-5: Dijkstra)
"""
import unittest
from egx.intelligence.graph.topology_graph import HardwareTopologyGraph

class TestDijkstra(unittest.TestCase):
    def test_shortest_path(self):
        g = HardwareTopologyGraph()
        g.add_edge("gpu0", "gpu1", 100.0)
        g.add_edge("gpu1", "gpu2", 50.0)
        g.add_edge("gpu0", "gpu2", 10.0)
        path, cost = g.shortest_path("gpu0", "gpu2")
        self.assertIn("gpu2", path)

if __name__ == "__main__":
    unittest.main()
