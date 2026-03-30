"""Deterministic mock batch iterator for EGX test suite."""

import torch
from typing import Dict, List, Any


class MockBatchIterator:
    """Yields deterministic batches without needing a real dataset."""

    def __init__(
        self,
        batch_size: int = 4,
        seq_len: int = 32,
        vocab_size: int = 1000,
        num_batches: int = 5,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            yield {
                "input_ids": torch.randint(
                    0, self.vocab_size, (self.batch_size, self.seq_len)
                ),
                "attention_mask": torch.ones(
                    self.batch_size, self.seq_len, dtype=torch.long
                ),
                "labels": torch.randint(
                    0, self.vocab_size, (self.batch_size, self.seq_len)
                ),
            }

    def __len__(self):
        return self.num_batches


class MockDataset(torch.utils.data.Dataset):
    """Minimal in-memory dataset for DataLoader tests."""

    def __init__(self, size: int = 100, seq_len: int = 32):
        self.data = [
            {
                "input_ids": torch.randint(0, 1000, (seq_len,)),
                "attention_mask": torch.ones(seq_len, dtype=torch.long),
            }
            for _ in range(size)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
