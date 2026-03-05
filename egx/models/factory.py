"""
EGX Model Factory — Layer 5.

Initializes models from scratch or loads them from configs.
Supports "Training from Scratch" as a core capability.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

logger = logging.getLogger("egx.models")

class ModelFactory:
    """
    EGX Model Initialization.
    Supports creating fresh models for scratch training.
    """

    @staticmethod
    def create_from_config(config: Any) -> Any:
        """
        Creates a model with random initialization based on an arch config.
        """
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for ModelFactory.")
            
        logger.info("Models: Initializing fresh model from scratch...")
        
        # Example: if it's a HuggingFace config, we use AutoModel
        # For now, let's create a simple MLP if it's a custom dict
        if isinstance(config, dict):
            return ModelFactory._create_simple_mlp(config)
            
        # Fallback to standard HF pattern if it looks like one
        try:
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM.from_config(config)
        except ImportError:
            logger.warning("Transformers not found, cannot initialize from HF config.")
        
        return None

    @staticmethod
    def _create_simple_mlp(config: Dict[str, Any]) -> Any:
        dims = config.get("dims", [784, 128, 10])
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    @staticmethod
    def get_tiny_test_model() -> Any:
        """Helper for laptop-based testing."""
        return ModelFactory._create_simple_mlp({"dims": [10, 10, 10]})
