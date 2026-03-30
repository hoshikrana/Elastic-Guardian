"""Unit tests for SafeTensorsExporter."""

import unittest


class TestSafeTensorsExporter(unittest.TestCase):
    def test_format_name(self):
        from egx.export.safetensors_exporter import SafeTensorsExporter

        self.assertEqual(SafeTensorsExporter().format_name, "SafeTensors")

    def test_inherits_base(self):
        from egx.export.safetensors_exporter import SafeTensorsExporter
        from egx.export.base_exporter import BaseExporter

        self.assertTrue(issubclass(SafeTensorsExporter, BaseExporter))


if __name__ == "__main__":
    unittest.main()
