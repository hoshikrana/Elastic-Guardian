"""
EGX HuggingFace Dataset Adapter — Layer 5.

Lightweight wrapper to use HuggingFace datasets as PyTorch datasets
without materializing everything into a list of tensors.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object

logger = logging.getLogger("egx.data.hf_adapter")


class HFDatasetAdapter(Dataset):
    """
    Wraps a HuggingFace Dataset (or any Arrow-backed dataset) into a
    PyTorch Dataset that lazily converts rows to tensors on-the-fly.

    This avoids the massive memory spike of converting all 50K+ rows
    to tensors upfront.

    Usage:
        from datasets import load_dataset
        raw = load_dataset("yahma/alpaca-cleaned", split="train")

        # Tokenize with .map() first
        tokenized = raw.map(tokenize_fn, batched=True, remove_columns=raw.column_names)

        # Wrap into PyTorch Dataset
        train_dataset = HFDatasetAdapter(tokenized, tensor_columns=["input_ids", "attention_mask", "labels"])
    """

    def __init__(
        self,
        hf_dataset: Any,
        tensor_columns: Optional[list] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            hf_dataset: A HuggingFace datasets.Dataset object.
            tensor_columns: Columns to convert to torch.Tensor.
                           If None, converts all list/int columns.
            transform: Optional function applied to each sample dict
                      after tensor conversion.
        """
        self.dataset = hf_dataset
        self.tensor_columns = tensor_columns or [
            "input_ids",
            "attention_mask",
            "labels",
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        result = {}

        for key, value in item.items():
            if key in self.tensor_columns:
                if torch is not None and not isinstance(value, torch.Tensor):
                    result[key] = torch.tensor(value)
                else:
                    result[key] = value
            else:
                result[key] = value

        if self.transform:
            result = self.transform(result)

        return result

    @classmethod
    def from_tokenized(
        cls,
        hf_dataset: Any,
        tokenizer: Any,
        text_column: str = "text",
        max_length: int = 512,
        **kwargs,
    ) -> "HFDatasetAdapter":
        """
        Convenience factory: tokenize + adapt in one call.

        Args:
            hf_dataset: Raw HuggingFace dataset.
            tokenizer: HuggingFace tokenizer.
            text_column: Name of the column containing text.
            max_length: Maximum sequence length.
        """

        def tokenize_fn(batch):
            tokens = tokenizer(
                batch[text_column],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens

        tokenized = hf_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=hf_dataset.column_names,
            desc="Tokenizing dataset",
        )

        return cls(tokenized, **kwargs)
