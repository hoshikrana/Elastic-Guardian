"""Integration: Zero-config boot → train → result."""

import unittest


class TestZeroConfig(unittest.TestCase):
    def test_default_boot_and_train(self):
        from egx.api.trainer import EGX
        from tests.mocks.mock_model import TinyModel

        trainer = EGX()
        model = TinyModel()
        result = trainer.train(model=model, dataset=[])
        self.assertTrue(result["success"])
        self.assertIn("final_loss", result)

    def test_dict_config_boot(self):
        from egx.api.trainer import EGX
        from tests.mocks.mock_model import TinyModel

        trainer = EGX({"num_epochs": 1, "learning_rate": 0.01})
        result = trainer.train(model=TinyModel(), dataset=[])
        self.assertTrue(result["success"])

    def test_config_object_boot(self):
        from egx.api.config import EGXConfig
        from egx.api.trainer import EGX
        from tests.mocks.mock_model import TinyModel

        cfg = EGXConfig(num_epochs=2)
        trainer = EGX(cfg)
        result = trainer.train(model=TinyModel(), dataset=[])
        self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()
