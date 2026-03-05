"""
EGX Test: resilience/watchdog.py
"""
import unittest
from egx.resilience.watchdog import TrainingWatchdog

class TestWatchdog(unittest.TestCase):
    def test_heartbeat(self):
        wd = TrainingWatchdog(timeout_s=10.0)
        wd.heartbeat(1)
        wd.heartbeat(2)
        # No exception = success

if __name__ == "__main__":
    unittest.main()
