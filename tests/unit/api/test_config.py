"""
EGX Test: api/config.py
"""

import unittest
from egx.api.config import EGXConfig


class TestAPIConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = EGXConfig()
        self.assertEqual(cfg.num_epochs, 3)
        self.assertEqual(cfg.lora_rank, 16)

    def test_from_dict(self):
        cfg = EGXConfig.from_dict({"num_epochs": 10, "custom_key": "v"})
        self.assertEqual(cfg.num_epochs, 10)
        self.assertEqual(cfg.get("custom_key"), "v")


if __name__ == "__main__":
    unittest.main()
