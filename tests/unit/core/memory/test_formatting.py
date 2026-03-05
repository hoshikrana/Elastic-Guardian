"""
EGX Test: core/memory/formatting.py
"""
import unittest
from egx.core.memory.formatting import format_bytes

class TestMemoryFormatting(unittest.TestCase):
    def test_format_bytes_gb(self):
        self.assertIn("GB", format_bytes(1024**3))
    
    def test_format_bytes_mb(self):
        self.assertIn("MB", format_bytes(1024**2))

if __name__ == "__main__":
    unittest.main()
