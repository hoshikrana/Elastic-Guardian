"""
EGX Core Interfaces — Deep Architectural Contract.

Defines the explicit Abstract Base Classes (ABCs) required to achieve 
true Polymorphism and Dependency Inversion in the EGX runtime.
Developer extension plugins must inherit these interfaces.
"""

from __future__ import annotations
 
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from egx.core.models import (
    GPUSpec, 
    HardwareTopology, 
    MemoryReport, 
    ModelProfile, 
    TrainingPlan
)

class BaseGPUProber(ABC):
    """Architectural contract for hardware enumeration."""

    @abstractmethod
    def probe(self) -> List[GPUSpec]:
        """Probes the physical hardware and returns a list of specifications."""
        pass

    @abstractmethod
    def __enter__(self) -> BaseGPUProber:
        """Initialize hardware context."""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Shutdown hardware context and release resources."""
        pass


class BaseTopologyBuilder(ABC):
    """Architectural contract for topology assembly."""
    
    @abstractmethod
    def build(self, gpus: List[GPUSpec]) -> HardwareTopology:
        """Assembles isolated GPU specs into a unified logical topology."""
        pass


class BaseTrainingKernel(ABC):
    """
    Architectural contract for the stateless mathematical execution kernel.
    """
    
    @abstractmethod
    def train_step(self, batch: Dict[str, Any], step: int) -> float:
        """Executes a single secure forward/backward training step."""
        pass


class BaseEngine(ABC):
    """
    Architectural contract for the primary EGX lifecycle orchestrator.
    """
    
    @abstractmethod
    def boot(self, model: Any) -> None:
        """Executes hardware alignment Phases 1-4."""
        pass

    @abstractmethod
    def run_training(
        self,
        model: Any,
        dataset: Any,
        eval_dataset: Optional[Any],
        config: Any,
        **kwargs,
    ) -> Dict[str, Any]:
        """Executes the core mathematical runtime Phases 5-10."""
        pass

class BaseCheckpointManager(ABC):
    """Contract for adaptive checkpoint lifecycle logic."""
    @abstractmethod
    def should_save(self, step: int, loss: float) -> bool: pass
    
    @abstractmethod
    def checkpoint(self, step: int, loss: float, state_dict: dict): pass

class BaseWatchdog(ABC):
    """Contract for heartbeat deadlock detection."""
    @abstractmethod
    def start(self): pass
    
    @abstractmethod
    def stop(self): pass
    
    @abstractmethod
    def heartbeat(self, step: int): pass

class BaseStrategySelector(ABC):
    """Contract for selecting optimal structural strategies during Phase 5."""
    @abstractmethod
    def insert(self, priority: float, value: Any): pass
    
    @abstractmethod
    def extract_max(self) -> Any: pass


class BaseEstimator(ABC):
    """Architectural contract for structural memory estimation."""

    @abstractmethod
    def estimate(
        self, topology: HardwareTopology, profile: ModelProfile, plan: TrainingPlan
    ) -> MemoryReport:
        """Calculates estimated memory requirements."""
        pass

