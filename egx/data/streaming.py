"""
EGX Elastic Dataset — Layer 5.

Streaming dataset implementation for training on massive data
without loading everything into RAM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Union

import torch
from torch.utils.data import IterableDataset


class ElasticDataset(IterableDataset):
    """
    Streaming implementation for Layer 7 zero-config training.
    Supports JSONL and Safetensors shard streaming.
    """

    def __init__(
        self,
        data_path: Union[str, Path, List[str]],
        tokenizer: Any,
        max_seq_len: int = 2048,
        infinite_loop: bool = False,
    ):
        self.paths = [
            Path(p)
            for p in ([data_path] if isinstance(data_path, (str, Path)) else data_path)
        ]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.infinite_loop = infinite_loop

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Yields tokenized features from the streaming source.
        """
        while True:
            for path in self.paths:
                if path.suffix == ".jsonl":
                    yield from self._stream_jsonl(path)
                elif path.suffix == ".txt":
                    yield from self._stream_text(path)
                else:
                    raise ValueError(f"Unsupported data format: {path.suffix}")

            if not self.infinite_loop:
                break

    def _stream_jsonl(self, path: Path) -> Iterator[Dict[str, torch.Tensor]]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                text = data.get("text") or data.get("content") or ""
                if not text:
                    continue

                features = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_seq_len,
                    return_tensors="pt",
                )

                yield {k: v.squeeze(0) for k, v in features.items()}

    def _stream_text(self, path: Path) -> Iterator[Dict[str, torch.Tensor]]:
        with open(path, "r", encoding="utf-8") as f:
            # Simple chunked text streaming
            chunk_size = 1024 * 1024  # 1MB
            while True:
                content = f.read(chunk_size)
                if not content:
                    break

                features = self.tokenizer(
                    content,
                    truncation=True,
                    max_length=self.max_seq_len,
                    return_tensors="pt",
                )
                yield {k: v.squeeze(0) for k, v in features.items()}
