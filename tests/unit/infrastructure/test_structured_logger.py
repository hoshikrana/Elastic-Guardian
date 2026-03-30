"""
EGX Test: infrastructure/structured_logger.py
"""

import unittest
from egx.infrastructure.structured_logger import StructuredLogger


class TestStructuredLogger(unittest.TestCase):
    def test_log_event(self):
        logger = StructuredLogger("test")
        # Should not raise
        logger.log_event("test_event", {"key": "value"})

    def test_log_degradation(self):
        logger = StructuredLogger("test")
        logger.log_degradation("gpu", "thermal", "throttle")


if __name__ == "__main__":
    unittest.main()
