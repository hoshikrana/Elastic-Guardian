"""Unit tests for LifecycleManager."""
import unittest


class TestLifecycleManager(unittest.TestCase):
    def test_initial_phase(self):
        from egx.runtime.lifecycle import LifecycleManager, LifecyclePhase
        lm = LifecycleManager()
        self.assertEqual(lm.current_phase, LifecyclePhase.PROBE)

    def test_transition(self):
        from egx.runtime.lifecycle import LifecycleManager, LifecyclePhase
        lm = LifecycleManager()
        lm.transition_to(LifecyclePhase.PLAN)
        self.assertEqual(lm.current_phase, LifecyclePhase.PLAN)
        self.assertIn(LifecyclePhase.PROBE, lm.completed_phases)

    def test_is_ready_first_phase(self):
        from egx.runtime.lifecycle import LifecycleManager, LifecyclePhase
        lm = LifecycleManager()
        self.assertTrue(lm.is_ready(LifecyclePhase.PROBE))

    def test_is_ready_requires_previous(self):
        from egx.runtime.lifecycle import LifecycleManager, LifecyclePhase
        lm = LifecycleManager()
        # PLAN (idx=1) requires PROBE (idx=0) in completed_phases
        # PROBE is current_phase but not yet in completed_phases
        self.assertFalse(lm.is_ready(LifecyclePhase.PLAN))
        # Transition: PROBE -> completed, current -> PLAN
        lm.transition_to(LifecyclePhase.PLAN)
        self.assertIn(LifecyclePhase.PROBE, lm.completed_phases)
        # Now check: is PLAN ready? Its predecessor PROBE is in completed_phases
        self.assertTrue(lm.is_ready(LifecyclePhase.PLAN))


if __name__ == "__main__":
    unittest.main()
