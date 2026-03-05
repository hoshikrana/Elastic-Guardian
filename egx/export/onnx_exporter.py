"""
EGX ONNX Exporter — Layer 5.

Exports models to ONNX format for cross-platform inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from egx.export.base_exporter import BaseExporter

logger = logging.getLogger("egx.export.onnx")


class ONNXExporter(BaseExporter):
    """Exports PyTorch models to ONNX format."""

    @property
    def format_name(self) -> str:
        return "ONNX"

    def export(
        self,
        model: Any,
        output_path: Path,
        opset_version: int = 17,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        input_names: Optional[list] = None,
        output_names: Optional[list] = None,
        **kwargs,
    ) -> Path:
        output_path = self._ensure_dir(Path(output_path))
        model.eval()

        # Default dummy input for language models
        dummy = torch.zeros(1, 128, dtype=torch.long)
        if torch.cuda.is_available():
            dummy = dummy.cuda()
            model = model.cuda()

        if dynamic_axes is None:
            dynamic_axes = {
                "input_ids": {0: "batch", 1: "seq"},
                "output": {0: "batch", 1: "seq"},
            }
        if input_names is None:
            input_names = ["input_ids"]
        if output_names is None:
            output_names = ["output"]

        torch.onnx.export(
            model,
            (dummy,),
            str(output_path),
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        logger.info(f"ONNX export complete: {output_path}")
        return output_path

    def validate(self, output_path: Path) -> bool:
        try:
            import onnx
            model = onnx.load(str(output_path))
            onnx.checker.check_model(model)
            logger.info(f"ONNX validation passed: {output_path}")
            return True
        except Exception as e:
            logger.error(f"ONNX validation failed: {e}")
            return False
