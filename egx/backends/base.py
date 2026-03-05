"""
EGX Backend Base — Layer 5.

Abstract interface for training framework adapters (PyTorch, JAX, etc.).
Ensures EGX core remains framework-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from egx.core.memory.value import MemoryValue


class TrainingBackend(ABC):
    """
    EGX Backend Interface.
    Framework-specific logic (e.g. PyTorch) goes into subclasses.
    """

    @abstractmethod
    def get_device_memory_usage(self, device_id: str) -> MemoryValue:
        """Query framework for its view of memory usage."""
        pass

    @abstractmethod
    def setup_model(self, model: Any, strategy: Any) -> Any:
        """Perform initial model wrapping/injection."""
        pass

    @abstractmethod
    def train_step(self, model: Any, batch: Any) -> Dict[str, Any]:
        """Execute a single training iteration."""
        pass

    @property
    @abstractmethod
    def framework_name(self) -> str:
        return "base"
