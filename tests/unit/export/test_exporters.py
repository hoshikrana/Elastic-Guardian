"""Unit tests for ONNX and SafeTensors exporters."""

import unittest


class TestONNXExporter(unittest.TestCase):
    def test_format_name(self):
        from egx.export.onnx_exporter import ONNXExporter

        self.assertEqual(ONNXExporter().format_name, "ONNX")


class TestSafeTensorsExporter(unittest.TestCase):
    def test_format_name(self):
        from egx.export.safetensors_exporter import SafeTensorsExporter

        self.assertEqual(SafeTensorsExporter().format_name, "SafeTensors")


class TestLoRAExportMerger(unittest.TestCase):
    def test_merge_and_export_returns_path(self):
        from egx.export.lora_merger import LoRAExportMerger
        import torch.nn as nn

        merger = LoRAExportMerger()
        path = merger.merge_and_export(nn.Linear(16, 8), "/tmp/test_export")
        self.assertEqual(path, "/tmp/test_export")


if __name__ == "__main__":
    unittest.main()
