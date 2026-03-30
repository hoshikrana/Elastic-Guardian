"""Unit tests for CheckpointWriter."""

import unittest
import tempfile
import os
import torch


class TestCheckpointWriter(unittest.TestCase):
    def test_atomic_save(self):
        from egx.resilience.checkpoint.writer import CheckpointWriter

        writer = CheckpointWriter()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            writer.save({"step": 1}, path)
            self.assertTrue(os.path.exists(path))

    def test_sha256_sidecar(self):
        from egx.resilience.checkpoint.writer import CheckpointWriter

        writer = CheckpointWriter()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            writer.save({"step": 1, "w": torch.randn(5)}, path)
            sha_path = path.replace(".pt", ".sha256")
            self.assertTrue(os.path.exists(sha_path))
            with open(sha_path) as f:
                self.assertEqual(len(f.read()), 64)

    def test_no_tmp_file_left_on_failure(self):
        from egx.resilience.checkpoint.writer import CheckpointWriter

        writer = CheckpointWriter()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            writer.save({"x": torch.randn(3)}, path)
            tmp = path.replace(".pt", ".tmp")
            self.assertFalse(os.path.exists(tmp))


if __name__ == "__main__":
    unittest.main()
