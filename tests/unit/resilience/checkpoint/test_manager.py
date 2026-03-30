"""
EGX Test: resilience/checkpoint/manager.py
"""

import unittest
import tempfile
from egx.resilience.checkpoint.manager import CheckpointManager
from egx.core.enums import CheckpointStrategy


class TestCheckpointManager(unittest.TestCase):
    def test_should_save_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                output_dir=tmpdir, strategy=CheckpointStrategy.STEP_BASED
            )
            self.assertFalse(mgr.should_save(5, 0.5))
            self.assertTrue(mgr.should_save(500, 0.5))

    def test_should_save_loss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                output_dir=tmpdir, strategy=CheckpointStrategy.LOSS_BASED
            )
            mgr.checkpoint(1, 1.0, {})  # initial
            self.assertTrue(mgr.should_save(2, 0.8))  # > 0.99


if __name__ == "__main__":
    unittest.main()
