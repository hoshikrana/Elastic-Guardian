"""
EGX Test: resilience/checkpoint/reader.py
"""

import unittest
import os
import tempfile
import torch
from egx.resilience.checkpoint.writer import CheckpointWriter
from egx.resilience.checkpoint.reader import CheckpointReader


class TestCheckpointReader(unittest.TestCase):
    def test_read_verified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CheckpointWriter()
            data = {"step": 1, "loss": 0.5, "data": {"weights": torch.randn(10)}}
            path = os.path.join(tmpdir, "checkpoint.pt")
            writer.save(data, path)

            reader = CheckpointReader()
            loaded = reader.load(path)
            self.assertEqual(loaded["step"], 1)
            self.assertIn("weights", loaded["data"])


if __name__ == "__main__":
    unittest.main()
