"""
EGX Model Introspector — Layer 5.

Analyzes nn.Module to produce ModelProfile.
Detects layers, hidden dimensions, and parameter counts.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("egx.models")

class ModelProfile:
    """The static footprint of a model."""
    def __init__(
        self, 
        name: str, 
        params: int, 
        hidden_dim: int, 
        layers: int, 
        arch: str
    ):
        self.name = name
        self.params = params
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.arch = arch

class ModelIntrospector:
    """
    EGX Introspection.
    Extracts deep metadata from PyTorch models.
    """

    def introspect(self, model: Any, name: str = "unknown") -> ModelProfile:
        """
        Walks the module tree to identify transformer components.
        """
        logger.info(f"Models: Analyzing architecture of '{name}'...")
        
        # 1. Parameter count
        params = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else 0
        
        # 2. Heuristic-based dimension detection
        hidden_dim = 0
        layers = 0
        arch = "unknown"
        
        # Attempt to find hidden_size/config
        if hasattr(model, "config"):
            config = model.config
            hidden_dim = getattr(config, "hidden_size", 0) or getattr(config, "d_model", 0)
            layers = getattr(config, "num_hidden_layers", 0) or getattr(config, "n_layer", 0)
            arch = getattr(config, "model_type", "transformer")
            
        # 3. Fallback to module counting if no config
        if layers == 0 and hasattr(model, "modules"):
            # Count modules with 'layer' or 'block' in their name
            layers = sum(1 for name, mod in model.named_modules() if "layer" in name.lower() or "block" in name.lower())

        return ModelProfile(
            name=name,
            params=params, 
            hidden_dim=hidden_dim, 
            layers=layers, 
            arch=arch
        )
