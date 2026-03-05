"""
EGX Estimator Base — Layer 3.

Defines the interface for all memory estimation engines.
Rules:
  - No torch imports (L3 boundaries)
  - All results must be MemoryReport (Law 10)
  - Must be thread-safe for parallel simulation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from egx.core.models import (
        HardwareTopology,
        ModelProfile,
        MemoryReport,
        TrainingPlan,
    )


class BaseEstimator(ABC):
    """
    Abstract base for memory estimators.

    Subclasses implement different fidelity levels:
    - Analytical: Formula-based, <1ms.
    - DryRun: Measured on-device, 5-30s.
    - Hybrid: Analytical with measured calibration.
    """

    @abstractmethod
    def estimate(
        self, topology: HardwareTopology, profile: ModelProfile, plan: TrainingPlan
    ) -> MemoryReport:
        """
        Produce a MemoryReport for the given configuration.
        Should never raise for 'out of memory' — instead return a report
        where total_bytes > available_vram.
        """
        pass
