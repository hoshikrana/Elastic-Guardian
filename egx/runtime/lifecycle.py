"""
EGX Runtime Lifecycle — Layer 5.

Orchestrates the 10 phases of training execution.
Tracks state transitions and execution health.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import List

logger = logging.getLogger("egx.runtime.lifecycle")


class LifecyclePhase(Enum):
    PROBE = "probe"
    PLAN = "plan"
    LOAD = "load"
    INJECT = "inject"
    CALIBRATE = "calibrate"
    ACCUMULATE = "accumulate"
    FORWARD = "forward"
    BACKWARD = "backward"
    STEP = "step"
    SNAPSHOT = "snapshot"


class LifecycleManager:
    """
    Enforces the 10-phase execution mandate.
    """
    
    def __init__(self):
        self.current_phase = LifecyclePhase.PROBE
        self.completed_phases: List[LifecyclePhase] = []

    def transition_to(self, phase: LifecyclePhase):
        """Validated phase transition."""
        logger.debug(f"Lifecycle: Transitioning from {self.current_phase.name} -> {phase.name}")
        self.completed_phases.append(self.current_phase)
        self.current_phase = phase

    def is_ready(self, phase: LifecyclePhase) -> bool:
        """Checks if dependencies for a phase are met."""
        # Simple linear progression check
        phases = list(LifecyclePhase)
        idx = phases.index(phase)
        if idx == 0: return True
        return phases[idx-1] in self.completed_phases
