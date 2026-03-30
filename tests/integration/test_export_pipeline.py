"""Integration: Model → export → validate."""

import unittest
import tempfile
import torch.nn as nn


class TestExportPipeline(unittest.TestCase):
    def test_lora_merger_end_to_end(self):
        from egx.export.lora_merger import LoRAExportMerger

        model = nn.Linear(32, 16)
        merger = LoRAExportMerger()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = merger.merge_and_export(model, f"{tmpdir}/exported")
            self.assertIsNotNone(path)

    def test_onnx_exporter_format_name(self):
        from egx.export.onnx_exporter import ONNXExporter

        exp = ONNXExporter()
        self.assertEqual(exp.format_name, "ONNX")

    def test_safetensors_exporter_format_name(self):
        from egx.export.safetensors_exporter import SafeTensorsExporter

        exp = SafeTensorsExporter()
        self.assertEqual(exp.format_name, "SafeTensors")

    def test_base_exporter_is_abstract(self):
        from egx.export.base_exporter import BaseExporter

        with self.assertRaises(TypeError):
            BaseExporter()


if __name__ == "__main__":
    unittest.main()
