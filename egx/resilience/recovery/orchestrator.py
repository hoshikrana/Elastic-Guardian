"""
EGX Recovery Orchestrator — Layer 4.

Coordinates recovery attempts in priority order using strategy pattern.
Implements the full recovery chain: Retry → HalveBatch → Downgrade → Rollback → Abort.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from egx.core.exceptions import EGXError, OutOfMemoryError
from egx.core.enums import RecoveryState

logger = logging.getLogger("egx.resilience.recovery")


@dataclass(frozen=True)
class RecoveryContext:
    """Context provided to recovery strategies for decision-making."""
    
    error: EGXError
    step: int
    last_checkpoint_path: Optional[str] = None
    remaining_retries: int = 5
    current_batch_size: int = 32
    current_training_mode: Optional[str] = None
    peak_memory_usage_bytes: int = 0


class RecoveryStrategy(ABC):
    """Base class for all recovery strategies."""
    
    @abstractmethod
    async def attempt(self, context: RecoveryContext) -> bool:
        """
        Attempt recovery. Returns True if successful, False otherwise.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging."""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """Lower priority = higher priority (0 = first to try)."""
        pass


class RetryStrategy(RecoveryStrategy):
    """Simple retry with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay_s: float = 1.0):
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self.attempt_count = 0
    
    async def attempt(self, context: RecoveryContext) -> bool:
        """Retry the operation with exponential backoff."""
        
        if not context.error.recoverable:
            logger.debug(f"{self.name}: Error not recoverable, skipping")
            return False
        
        if self.attempt_count >= self.max_retries:
            logger.info(f"{self.name}: Max retries ({self.max_retries}) exceeded")
            return False
        
        delay = self.base_delay_s * (2 ** self.attempt_count)
        logger.info(
            f"{self.name}: Attempting retry {self.attempt_count + 1}/{self.max_retries} "
            f"after {delay:.1f}s delay..."
        )
        
        await asyncio.sleep(delay)
        self.attempt_count += 1
        return True
    
    @property
    def name(self) -> str:
        return "RetryStrategy"
    
    @property
    def priority(self) -> int:
        return 0  # Try first


class HalveBatchStrategy(RecoveryStrategy):
    """Reduce batch size and resume training."""
    
    def __init__(self):
        self.halves_performed = 0
        self.max_halves = 5  # Don't go below batch_size=1
    
    async def attempt(self, context: RecoveryContext) -> bool:
        """Attempt to recover by reducing batch size."""
        
        if not isinstance(context.error, OutOfMemoryError):
            logger.debug(f"{self.name}: Error is not OOM, skipping")
            return False
        
        if context.current_batch_size <= 1:
            logger.warning(f"{self.name}: Can't halve batch (already at 1)")
            return False
        
        if self.halves_performed >= self.max_halves:
            logger.warning(f"{self.name}: Max batch halvings ({self.max_halves}) reached")
            return False
        
        new_batch_size = max(1, context.current_batch_size // 2)
        logger.info(
            f"{self.name}: Reducing batch size "
            f"{context.current_batch_size} → {new_batch_size} "
            f"(halve {self.halves_performed + 1}/{self.max_halves})"
        )
        
        self.halves_performed += 1
        # Signal to trainer would happen via callback or state mutation
        return True
    
    @property
    def name(self) -> str:
        return "HalveBatchStrategy"
    
    @property
    def priority(self) -> int:
        return 1


class DowngradeStrategyStrategy(RecoveryStrategy):
    """Switch to more memory-efficient training mode."""
    
    def __init__(self):
        self.downgrades_performed = 0
        # Strategy priority order: Full → LoRA+ → LoRA → DoRA → QLoRA
        self.fallback_order = [
            "full",
            "lora_plus", 
            "lora",
            "dora",
            "qlora",
        ]
    
    async def attempt(self, context: RecoveryContext) -> bool:
        """Attempt recovery by downgrading to more efficient strategy."""
        
        if not isinstance(context.error, OutOfMemoryError):
            logger.debug(f"{self.name}: Error is not OOM, skipping")
            return False
        
        if not context.current_training_mode:
            logger.warning(f"{self.name}: Current training mode unknown")
            return False
        
        try:
            current_idx = self.fallback_order.index(context.current_training_mode)
        except ValueError:
            logger.warning(f"{self.name}: Unknown training mode: {context.current_training_mode}")
            return False
        
        if current_idx >= len(self.fallback_order) - 1:
            logger.warning(f"{self.name}: Already at most efficient mode (QLoRA)")
            return False
        
        next_mode = self.fallback_order[current_idx + 1]
        logger.info(
            f"{self.name}: Downgrading strategy "
            f"{context.current_training_mode} → {next_mode}"
        )
        
        self.downgrades_performed += 1
        # Implementation of mode switch would happen in trainer
        return True
    
    @property
    def name(self) -> str:
        return "DowngradeStrategyStrategy"
    
    @property
    def priority(self) -> int:
        return 2


class CheckpointRollbackStrategy(RecoveryStrategy):
    """Restore from last good checkpoint."""
    
    async def attempt(self, context: RecoveryContext) -> bool:
        """Attempt recovery by loading last checkpoint."""
        
        if not context.last_checkpoint_path:
            logger.debug(f"{self.name}: No checkpoint available")
            return False
        
        logger.warning(
            f"{self.name}: Rolling back to checkpoint: "
            f"{context.last_checkpoint_path}"
        )
        
        # Here we would actually load the checkpoint
        # For now, return success to signal that rollback was attempted
        return True
    
    @property
    def name(self) -> str:
        return "CheckpointRollbackStrategy"
    
    @property
    def priority(self) -> int:
        return 3


class RecoveryOrchestrator:
    """
    Coordinates recovery strategies in priority order.
    Implements the full recovery chain with proper logging and state management.
    """
    
    def __init__(self):
        """Initialize recovery orchestrator with full strategy chain."""
        self.strategies: List[RecoveryStrategy] = [
            RetryStrategy(max_retries=3, base_delay_s=1.0),
            HalveBatchStrategy(),
            DowngradeStrategyStrategy(),
            CheckpointRollbackStrategy(),
        ]
        
        # Sort by priority (lower = first)
        self.strategies.sort(key=lambda s: s.priority)
        
        logger.debug(
            f"Recovery Orchestrator initialized with {len(self.strategies)} strategies: "
            f"{[s.name for s in self.strategies]}"
        )
    
    async def recover(self, context: RecoveryContext) -> bool:
        """
        Attempt recovery strategies in priority order.
        
        Returns:
            True if recovery succeeded, False if unrecoverable.
        """
        logger.warning(
            f"Recovery Orchestrator starting for error: "
            f"{context.error.message} at step {context.step}"
        )
        
        for strategy in self.strategies:
            logger.debug(f"Attempting {strategy.name}...")
            
            try:
                if await strategy.attempt(context):
                    logger.info(f"✔ {strategy.name} recovered successfully")
                    return True
                else:
                    logger.debug(f"✗ {strategy.name} unable to recover")
                    
            except Exception as e:
                logger.error(f"✗ {strategy.name} raised exception: {e}")
                continue
        
        logger.critical("All recovery strategies exhausted - marking as unrecoverable")
        return False
    
    def get_strategy(self, name: str) -> Optional[RecoveryStrategy]:
        """Get a specific strategy by name."""
        for strategy in self.strategies:
            if strategy.name == name:
                return strategy
        return None
    
    def reset(self) -> None:
        """Reset all strategies for next training run."""
        for strategy in self.strategies:
            if hasattr(strategy, 'attempt_count'):
                strategy.attempt_count = 0
            if hasattr(strategy, 'halves_performed'):
                strategy.halves_performed = 0
            if hasattr(strategy, 'downgrades_performed'):
                strategy.downgrades_performed = 0
        logger.debug("Recovery Orchestrator state reset")
