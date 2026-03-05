"""
EGX Test: core/exceptions.py
"""
import unittest
from egx.core.exceptions import (
    EGXError, HardwareError, GPUNotFoundError,
    EGXMemoryError, BoolAsIntError, OutOfMemoryError,
    NaNLossError, CircularDependencyError,
)
from egx.core.enums import RecoveryAction

class TestCoreExceptions(unittest.TestCase):
    def test_exception_hierarchy(self):
        self.assertTrue(issubclass(HardwareError, EGXError))
        self.assertTrue(issubclass(GPUNotFoundError, HardwareError))
        self.assertTrue(issubclass(BoolAsIntError, EGXMemoryError))

    def test_recoverability(self):
        oom = OutOfMemoryError()
        self.assertTrue(oom.recoverable)
        self.assertEqual(oom.suggested_action, RecoveryAction.HALVE_BATCH)
        
        gpu_err = GPUNotFoundError()
        self.assertFalse(gpu_err.recoverable)

    def test_nan_loss_includes_step(self):
        err = NaNLossError(step=42)
        self.assertIn("42", str(err))

    def test_circular_dependency(self):
        err = CircularDependencyError(cycle=["A", "B", "A"])
        self.assertIn("A", str(err))

if __name__ == "__main__":
    unittest.main()
