"""Unit tests for LoRAExportMerger."""
import unittest
import torch.nn as nn


class TestLoRAExportMerger(unittest.TestCase):
    def test_merge_returns_path(self):
        from egx.export.lora_merger import LoRAExportMerger
        merger = LoRAExportMerger()
        path = merger.merge_and_export(nn.Linear(16, 8), "/tmp/lora_export")
        self.assertEqual(path, "/tmp/lora_export")

    def test_merger_instantiation(self):
        from egx.export.lora_merger import LoRAExportMerger
        merger = LoRAExportMerger()
        self.assertIsNotNone(merger._merger)


if __name__ == "__main__":
    unittest.main()
