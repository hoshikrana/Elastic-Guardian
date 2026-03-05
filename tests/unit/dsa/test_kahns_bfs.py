"""
EGX Test: intelligence/graph/dependency_dag.py (DSA-6: Kahn's BFS)
"""
import unittest
from egx.intelligence.graph.dependency_dag import ModuleDependencyDAG
from egx.core.exceptions import CircularDependencyError

class TestKahnsAlgorithm(unittest.TestCase):
    def test_valid_dag(self):
        dag = ModuleDependencyDAG()
        dag.add_dependency("A", "B")
        dag.add_dependency("B", "C")
        order = dag.validate()
        self.assertIn("A", order)

    def test_cycle_detection(self):
        dag = ModuleDependencyDAG()
        dag.add_dependency("A", "B")
        dag.add_dependency("B", "C")
        dag.add_dependency("C", "A")
        with self.assertRaises(CircularDependencyError):
            dag.validate()

if __name__ == "__main__":
    unittest.main()
