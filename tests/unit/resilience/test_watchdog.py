"""Unit tests for TrainingWatchdog."""
import unittest


class TestTrainingWatchdog(unittest.TestCase):
    def test_heartbeat(self):
        from egx.resilience.watchdog import TrainingWatchdog
        wd = TrainingWatchdog(timeout_s=10.0)
        wd.heartbeat(1)
        wd.heartbeat(2)

    def test_inherits_base(self):
        from egx.resilience.watchdog import TrainingWatchdog
        from egx.core.interfaces import BaseWatchdog
        self.assertTrue(issubclass(TrainingWatchdog, BaseWatchdog))


if __name__ == "__main__":
    unittest.main()
