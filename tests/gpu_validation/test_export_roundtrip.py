"""GPU Validation: Export roundtrip tests."""
import unittest
import tempfile
import os
import torch
import torch.nn as nn


class TestExportRoundtrip(unittest.TestCase):
    def test_torch_save_load_roundtrip(self):
        model = nn.Linear(32, 16)
        original_weight = model.weight.clone()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            loaded = torch.load(f.name, weights_only=True)
        self.assertTrue(torch.allclose(original_weight, loaded["weight"]))
        os.unlink(f.name)

    def test_checkpoint_writer_roundtrip(self):
        from egx.resilience.checkpoint.writer import CheckpointWriter
        writer = CheckpointWriter()
        state = {"step": 42, "loss": 0.25, "state_dict": {"w": torch.randn(10)}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            writer.save(state, path)
            loaded = torch.load(path, weights_only=False)
            self.assertEqual(loaded["step"], 42)
            self.assertAlmostEqual(loaded["loss"], 0.25)


if __name__ == "__main__":
    unittest.main()
