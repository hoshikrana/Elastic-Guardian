"""
EGX Model Validator — Layer 6.

Provides strict validation of model weights and gradients to prevent
silent training failures (NaN/Inf).
"""

import torch
import logging

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Utility to validate model health before and during training.
    """

    __slots__ = ()

    @staticmethod
    def check_nans(model: torch.nn.Module) -> bool:
        """
        Checks all model parameters for NaN or Inf values.
        Returns True if the model is healthy, False otherwise.
        """
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                logger.error("NaN detected in parameter: %s", name)
                return False
            if torch.isinf(param).any():
                logger.error("Inf detected in parameter: %s", name)
                return False
        return True

    @staticmethod
    def check_gradients(model: torch.nn.Module) -> bool:
        """
        Checks all parameter gradients for NaN or Inf.
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    logger.error("NaN detected in gradient of: %s", name)
                    return False
                if torch.isinf(param.grad).any():
                    logger.error("Inf detected in gradient of: %s", name)
                    return False
        return True
