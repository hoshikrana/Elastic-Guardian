"""
EGX Plugin Registry — Layer 5.

Automated discovery and loading of hardware-specific optimizations.
"""

from __future__ import annotations

import logging
from typing import Dict


class PluginRegistry:
    """
    Registry for hardware-specific plugins (FlashAttn, ZeRO, etc.)
    """
    
    _plugins: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, plugin: Any):
        cls._plugins[name] = plugin
        logging.debug(f"Plugin: Registered {name}")

    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        return cls._plugins.get(name)

    @classmethod
    def initialize_all(cls):
        """Triggers activation for all supported plugins."""
        for name, plugin in cls._plugins.items():
            if hasattr(plugin, "is_supported") and plugin.is_supported():
                logging.info(f"Plugin: Activating {name}...")
                if hasattr(plugin, "apply"):
                    plugin.apply(None) # Context passed if needed
