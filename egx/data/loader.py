"""
EGX NVMe-Aware DataLoader — Layer 5.

Automatically tunes num_workers, pin_memory, and prefetch_factor
based on detected hardware topology.
"""

from __future__ import annotations

import multiprocessing
from typing import Optional, Union

from torch.utils.data import DataLoader, Dataset, IterableDataset
from egx.core.models import HardwareTopology


class NVMeDataLoader(DataLoader):
    """
    EGX Optimized DataLoader.
    Adapts concurrency to CPU core count and storage bandwidth.
    """

    def __init__(
        self,
        dataset: Union[Dataset, IterableDataset],
        batch_size: int = 1,
        shuffle: bool = False,
        topology: Optional[HardwareTopology] = None,
        **kwargs,
    ):
        # 1. Hardware-Aware Worker Selection
        cpu_count = multiprocessing.cpu_count()
        if topology:
            gpu_count = len(topology.gpus)
            suggested_workers = max(1, min(gpu_count * 2, cpu_count // 4))
        else:
            suggested_workers = max(1, cpu_count // 4)

        num_workers = kwargs.pop("num_workers", suggested_workers)

        # 2. Performance Flags
        pin_memory = kwargs.pop("pin_memory", True)
        prefetch_factor = kwargs.pop("prefetch_factor", 2)

        # 3. NVMe Optimization
        if topology and topology.nvme_seq_read_gbps > 2.0:
            prefetch_factor = max(prefetch_factor, 4)

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            **kwargs,
        )

        self.topology = topology
