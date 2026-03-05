"""
EGX Test: runtime/config_loader.py (DSA-8: Trie)
"""
import unittest
from egx.runtime.config_loader import ConfigTrie

class TestTrie(unittest.TestCase):
    def test_insert_get(self):
        t = ConfigTrie()
        t.insert("core.memory.limit", 1024)
        self.assertEqual(t.get("core.memory.limit"), 1024)
        self.assertIsNone(t.get("core.memory.nonexist"))

    def test_prefix_search(self):
        t = ConfigTrie()
        t.insert("core.a", 1)
        t.insert("core.b", 2)
        t.insert("other.c", 3)
        results = t.find_by_prefix("core")
        self.assertEqual(len(results), 2)

if __name__ == "__main__":
    unittest.main()
