"""
EGX Trainer — Layer 7.

The definitive v1.0 Public API for EGX.
Provides a high-level intersection for hardware awareness, data loaders,
PEFT injection, and resilient training loops.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

from egx.api.config import EGXConfig
from egx.runtime.engine import EGXEngine
from egx.core.exceptions import EGXError

logger = logging.getLogger("egx.api.trainer")


class EGXTrainer:
    """
    Law 12: Frozen API contract.
    The primary entry point for all EGX operations.
    """

    def __init__(self, config: Optional[Union[EGXConfig, Dict[str, Any]]] = None):
        """
        Initialize the trainer with optional configuration.
        Configures the underlying engine automatically.
        """
        if isinstance(config, dict):
            self.config = EGXConfig.from_dict(config)
        else:
            self.config = config or EGXConfig()

        self._engine = EGXEngine()
        self._is_booted = False

    def train(
        self,
        model: Any,
        dataset: Any,
        eval_dataset: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Main training entry point.
        Executes the 10-phase definitive lifecycle.
        """
        try:
            logger.info(
                f"EGX v1.0: Starting training session for {type(model).__name__}"
            )

            # 1. Boot the Engine (Phases 1-4)
            if not self._is_booted:
                self._engine.boot(model)
                self._is_booted = True

            # 2. Execute Training Logic (Phases 5-9)
            result = self._engine.run_training(
                model=model,
                dataset=dataset,
                eval_dataset=eval_dataset,
                config=self.config,
                **kwargs,
            )

            # 3. Shutdown (Phase 10)
            logger.info("EGX v1.0: Training session completed successfully.")
            return result

        except EGXError as e:
            logger.error(f"EGX Failure: {e.message}. Action: {e.suggested_action}")
            raise
        except Exception as e:
            logger.critical(f"Unexpected System Failure: {e}")
            raise EGXError(
                message="Fatal system error during training execution.",
                code="SYSTEM_CRITICAL",
                recoverable=False,
            ) from e


# Alias for definitive user access
EGX = EGXTrainer
