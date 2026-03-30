"""
EGX Model Loader — Layer 7.

Handles dynamic model loading from HuggingFace Hub or local paths.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

logger = logging.getLogger("egx.models.loader")


class AutoModelLoader:
    """
    Seamless model loading for HuggingFace Transformers.
    """

    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path: str,
        dtype: Optional[torch.dtype] = None,
        device_map: str = "auto",
        token: Optional[str] = None,
        load_tokenizer: bool = True,
        **kwargs,
    ) -> Tuple[Any, Optional[Any]]:
        """
        Loads a model and optionally its tokenizer.
        """
        if AutoModelForCausalLM is None:
            raise ImportError("transformers module is required for AutoModelLoader")

        logger.info(f"Loading model '{pretrained_model_name_or_path}'...")

        # Default to memory-efficient dtype
        dtype = dtype or torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=dtype,
            device_map=device_map,
            token=token,
            **kwargs,
        )

        tokenizer = None
        if load_tokenizer:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path,
                    token=token,
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            except Exception as e:
                logger.warning(
                    f"Failed to load tokenizer for {pretrained_model_name_or_path}: {e}"
                )

        logger.info("Model load complete.")
        return model, tokenizer
