"""
EGX Test: core/memory/validators.py
"""
import unittest
from egx.core.memory.validators import StandardMemoryValidator
from egx.core.exceptions import BoolAsIntError

class TestMemoryValidators(unittest.TestCase):
    def test_validate_int(self):
        self.assertEqual(StandardMemoryValidator.validate(1024, "test"), 1024)

    def test_validate_bool_trap(self):
        with self.assertRaises(BoolAsIntError):
            StandardMemoryValidator.validate(True, "test")

if __name__ == "__main__":
    unittest.main()
