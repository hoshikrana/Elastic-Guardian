"""Integration: Registry lookup → model config verification."""

import unittest


class TestModelLoading(unittest.TestCase):
    def test_builtin_model_lookup(self):
        from egx.models.registry import ModelRegistry

        reg = ModelRegistry()
        cfg = reg.get("llama2-7b")
        self.assertIsNotNone(cfg)
        self.assertGreater(cfg.param_count_approx, 0)

    def test_list_all_models(self):
        from egx.models.registry import ModelRegistry

        reg = ModelRegistry()
        available = reg.list_available()
        self.assertIsInstance(available, list)
        self.assertGreater(len(available), 0)

    def test_custom_model_registration(self):
        from egx.models.registry import ModelRegistry, ModelArchConfig

        reg = ModelRegistry()
        custom = ModelArchConfig("test-tiny", 512, 4, 4, 2048, 10000, 512, 100_000)
        reg.register(custom)
        self.assertIsNotNone(reg.get("test-tiny"))
        self.assertIn("test-tiny", reg.list_available())

    def test_unknown_model_returns_none(self):
        from egx.models.registry import ModelRegistry

        reg = ModelRegistry()
        self.assertIsNone(reg.get("nonexistent-model-xyz"))


if __name__ == "__main__":
    unittest.main()
