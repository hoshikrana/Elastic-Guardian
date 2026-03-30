"""
EGX Test: models/registry.py
"""

import unittest
from egx.models.registry import ModelRegistry, ModelArchConfig


class TestModelRegistry(unittest.TestCase):
    def test_builtin_lookup(self):
        reg = ModelRegistry()
        cfg = reg.get("llama2-7b")
        self.assertIsNotNone(cfg)
        self.assertGreater(cfg.param_count_approx, 0)

    def test_list_available(self):
        reg = ModelRegistry()
        available = reg.list_available()
        self.assertIn("llama2-7b", available)

    def test_custom_register(self):
        reg = ModelRegistry()
        custom = ModelArchConfig(
            "custom-1b", 2048, 24, 16, 8192, 32000, 2048, 1_000_000_000
        )
        reg.register(custom)
        self.assertIsNotNone(reg.get("custom-1b"))


if __name__ == "__main__":
    unittest.main()
