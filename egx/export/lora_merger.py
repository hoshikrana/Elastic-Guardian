"""
EGX Export Merger — Layer 5.

Orchestrates the weight merge and sharded safetensors saving.
"""

from __future__ import annotations

import logging
from typing import Any

from egx.peft.merger import LoRAMerger as PEFTWeightMerger

logger = logging.getLogger("egx.export")


class LoRAExportMerger:
    """
    High-level export engine.
    1. Merges PEFT weights.
    2. Optional quantization for deployment.
    3. Saves final artifact.
    """

    def __init__(self):
        self._merger = PEFTWeightMerger()

    def merge_and_export(self, model: Any, output_path: str):
        logger.info("Export: Preparing model for final delivery...")

        # 1. Merge adapters (Layer 5 merger)
        merged_model = self._merger.merge_weights(model)

        import os
        import torch
        from safetensors.torch import save_file

        # 2. Sharded export (safetensors)
        logger.info(f"Export: Saving sharded safetensors to '{output_path}'...")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        state_dict = merged_model.state_dict()
        tensors = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

        # 3. Precision Conversion (Optional FP8/INT4)
        logger.info("Export: Applying precision conversion to FP16...")
        for k, v in tensors.items():
            if v.dtype == torch.float32:
                tensors[k] = v.half()

        try:
            save_file(tensors, output_path)
            logger.info(f"✔ Model successfully exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            raise

        return output_path
