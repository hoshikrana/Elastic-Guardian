"""Integration: Train step → checkpoint save → verify file exists."""

import unittest
import tempfile
import os
import torch


class TestCheckpointPipeline(unittest.TestCase):
    def test_checkpoint_save_and_verify(self):
        from egx.resilience.checkpoint.manager import CheckpointManager
        from egx.core.enums import CheckpointStrategy

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                output_dir=tmpdir, strategy=CheckpointStrategy.STEP_BASED
            )
            state = {"weight": torch.randn(10), "step": 500}
            mgr.checkpoint(step=500, loss=0.3, state_dict=state)
            saved_file = os.path.join(tmpdir, "checkpoint_step_500.pt")
            self.assertTrue(os.path.exists(saved_file))

    def test_checkpoint_writer_atomicity(self):
        from egx.resilience.checkpoint.writer import CheckpointWriter

        writer = CheckpointWriter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_ckpt.pt")
            writer.save(
                {"step": 1, "loss": 0.5, "state_dict": {"w": torch.randn(5)}}, path
            )
            self.assertTrue(os.path.exists(path))
            sha_path = path.replace(".pt", ".sha256")
            self.assertTrue(os.path.exists(sha_path))
            with open(sha_path) as f:
                sha = f.read()
            self.assertEqual(len(sha), 64)

    def test_adaptive_should_save(self):
        from egx.resilience.checkpoint.manager import CheckpointManager

        mgr = CheckpointManager(output_dir="/tmp")
        # Initial best_loss is inf, so any loss < inf*0.99 triggers save
        self.assertTrue(mgr.should_save(step=1, loss=0.9))
        # After first save updates best_loss, marginal improvement should NOT trigger
        mgr._best_loss = 0.9
        self.assertFalse(mgr.should_save(step=2, loss=0.895))


if __name__ == "__main__":
    unittest.main()
