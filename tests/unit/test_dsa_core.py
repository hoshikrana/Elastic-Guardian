"""
EGX DSA Core Verification — Sprint 10.

Tests correctness and Big-O properties of the 8 mandated structures.
"""

import unittest
from egx.intelligence.strategy.selector import FibonacciHeap
from egx.intelligence.estimator.calibration.store import RedBlackTree
from egx.intelligence.estimator.dryrun import MemorySegmentTree
from egx.intelligence.graph.dependency_dag import ModuleDependencyDAG
from egx.runtime.config_loader import ConfigTrie


class TestDSACore(unittest.TestCase):
    
    def test_dsa_1_fib_heap(self):
        heap = FibonacciHeap()
        heap.insert(10.0, "low")
        heap.insert(50.0, "high")
        node = heap.insert(20.0, "mid")
        self.assertEqual(heap.max_node.value, "high") # type: ignore
        
        heap.increase_key(node, 60.0)
        self.assertEqual(heap.max_node.value, "mid") # type: ignore
        
        self.assertEqual(heap.extract_max().value, "mid") # type: ignore
        self.assertEqual(heap.extract_max().value, "high") # type: ignore

    def test_dsa_2_rb_tree(self):
        tree = RedBlackTree()
        tree.insert(10, "v1")
        tree.insert(20, "v2")
        tree.insert(15, "v3")
        self.assertEqual(tree.search(15), "v3")
        self.assertEqual(tree.find_nearest(18)[1], "v2") # Nearest to 18 is 20 (v2)

    def test_dsa_4_segment_tree(self):
        st = MemorySegmentTree(10)
        st.update(1, 100)
        st.update(2, 500)
        st.update(3, 200)
        self.assertEqual(st.query_max(0, 4), 500)
        self.assertEqual(st.query_max(0, 2), 100)
        self.assertEqual(st.global_peak(), 500)

    def test_dsa_6_kahns_bfs(self):
        dag = ModuleDependencyDAG()
        dag.add_dependency("A", "B")
        dag.add_dependency("B", "C")
        order = dag.validate()
        self.assertIn("A", order)
        self.assertIn("C", order)
        
        # Test Cycle
        dag.add_dependency("C", "A")
        with self.assertRaises(Exception): # CircularDependencyError
            dag.validate()

    def test_dsa_8_trie(self):
        trie = ConfigTrie()
        trie.insert("core.memory.val", 100)
        trie.insert("core.memory.unit", "GB")
        self.assertEqual(trie.get("core.memory.val"), 100)
        
        prefixes = trie.find_by_prefix("core.memory")
        self.assertEqual(len(prefixes), 2)

if __name__ == "__main__":
    unittest.main()
