"""
EGX Architecture Detector — Layer 5.

Identifies model architectures (Llama, Mistral, Bert, Custom)
from config files or weight signatures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from egx.core.enums import ArchType


class AutoArchDetector:
    """
    Intelligent architecture fingerprinting.
    """

    @staticmethod
    def detect(path: Union[str, Path]) -> ArchType:
        """
        Detects architecture type from model path.
        """
        path = Path(path)
        config_path = path / "config.json"

        if not config_path.exists():
            # Heuristic from folder name
            name = path.name.lower()
            if "llama" in name:
                return ArchType.LLAMA
            if "mistral" in name:
                return ArchType.MISTRAL
            return ArchType.PHANTOM  # Fallback

        with open(config_path, "r") as f:
            cfg = json.load(f)

        model_type = cfg.get("model_type", "").lower()
        if "llama" in model_type:
            return ArchType.LLAMA
        if "mistral" in model_type:
            return ArchType.MISTRAL
        if "falcon" in model_type:
            return ArchType.FALCON

        return ArchType.PHANTOM
