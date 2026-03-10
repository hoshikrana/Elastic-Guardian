"""Unit tests for BaseExporter ABC enforcement."""
import unittest


class TestBaseExporter(unittest.TestCase):
    def test_cannot_instantiate_abc(self):
        from egx.export.base_exporter import BaseExporter
        with self.assertRaises(TypeError):
            BaseExporter()

    def test_concrete_must_implement_export(self):
        from egx.export.base_exporter import BaseExporter
        class BadExporter(BaseExporter):
            @property
            def format_name(self): return "Bad"
            def validate(self, p): return True
        with self.assertRaises(TypeError):
            BadExporter()

    def test_ensure_dir_creates_parent(self):
        from egx.export.base_exporter import BaseExporter
        import tempfile, os
        from pathlib import Path
        class ConcreteExporter(BaseExporter):
            def export(self, model, path, **kw): return path
            def validate(self, p): return True
            @property
            def format_name(self): return "Test"
        e = ConcreteExporter()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sub" / "file.bin"
            result = e._ensure_dir(p)
            self.assertTrue(p.parent.exists())


if __name__ == "__main__":
    unittest.main()
