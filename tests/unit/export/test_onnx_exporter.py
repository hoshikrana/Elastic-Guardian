"""Unit tests for ONNXExporter."""
import unittest


class TestONNXExporter(unittest.TestCase):
    def test_format_name(self):
        from egx.export.onnx_exporter import ONNXExporter
        self.assertEqual(ONNXExporter().format_name, "ONNX")

    def test_inherits_base(self):
        from egx.export.onnx_exporter import ONNXExporter
        from egx.export.base_exporter import BaseExporter
        self.assertTrue(issubclass(ONNXExporter, BaseExporter))


if __name__ == "__main__":
    unittest.main()
