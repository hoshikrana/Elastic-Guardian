"""
EGX SafeTensors Exporter — Layer 5.

Exports model weights in SafeTensors format for safe, fast loading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import torch
from egx.export.base_exporter import BaseExporter

logger = logging.getLogger("egx.export.safetensors")


class SafeTensorsExporter(BaseExporter):
    """Exports model state_dict to SafeTensors format."""

    @property
    def format_name(self) -> str:
        return "SafeTensors"

    def export(self, model: Any, output_path: Path, **kwargs) -> Path:
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ImportError("safetensors is required: pip install safetensors")

        output_path = self._ensure_dir(Path(output_path))
        state_dict = model.state_dict() if hasattr(model, "state_dict") else model

        # SafeTensors requires all tensors on CPU
        cpu_state: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            cpu_state[k] = v.cpu().contiguous() if isinstance(v, torch.Tensor) else v

        save_file(cpu_state, str(output_path))
        logger.info(f"SafeTensors export complete: {output_path}")
        return output_path

    def validate(self, output_path: Path) -> bool:
        try:
            from safetensors import safe_open

            with safe_open(str(output_path), framework="pt") as f:
                keys = f.keys()
                logger.info(f"SafeTensors validation passed: {len(keys)} tensors")
                return len(keys) > 0
        except Exception as e:
            logger.error(f"SafeTensors validation failed: {e}")
            return False
