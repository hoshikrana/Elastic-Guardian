"""
EGX Test: resilience/checkpoint/writer.py
"""
import unittest
import os
import tempfile
import torch
from egx.resilience.checkpoint.writer import CheckpointWriter

class TestCheckpointWriter(unittest.TestCase):
    def test_atomic_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CheckpointWriter()
            path = os.path.join(tmpdir, "checkpoint.egx")
            data = {"weights": torch.randn(10)}
            writer.save(data, path)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(path.endswith(".egx"))

if __name__ == "__main__":
    unittest.main()
