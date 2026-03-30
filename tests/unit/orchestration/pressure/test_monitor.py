"""Unit tests for PressureEventSkipList."""

import unittest


class TestPressureEventSkipList(unittest.TestCase):
    def test_insert_and_latest(self):
        from egx.orchestration.pressure.monitor import PressureEventSkipList

        sl = PressureEventSkipList()
        sl.insert(1.0, "e1")
        sl.insert(5.0, "e5")
        sl.insert(3.0, "e3")
        self.assertEqual(sl.latest(), "e5")

    def test_find_at(self):
        from egx.orchestration.pressure.monitor import PressureEventSkipList

        sl = PressureEventSkipList()
        sl.insert(10.0, "event10")
        self.assertEqual(sl.find_at(10.0), "event10")
        self.assertIsNone(sl.find_at(99.0))

    def test_empty_latest(self):
        from egx.orchestration.pressure.monitor import PressureEventSkipList

        sl = PressureEventSkipList()
        self.assertIsNone(sl.latest())


if __name__ == "__main__":
    unittest.main()
