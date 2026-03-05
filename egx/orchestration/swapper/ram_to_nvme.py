"""
EGX RAM-to-NVMe Swapper — Layer 4.

Offloads tensors from system RAM to NVMe disk.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict
import torch

logger = logging.getLogger("egx.orchestration.swapper")


class RAMToNVMeSwapper:
    """Serializes tensors from RAM to disk for extreme offloading."""

    def __init__(self, cache_dir: str = ""):
        self._cache_dir = Path(cache_dir or tempfile.mkdtemp(prefix="egx_nvme_"))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest: Dict[str, Path] = {}

    def offload(self, name: str, tensor: torch.Tensor) -> int:
        """Save tensor to NVMe. Returns bytes written."""
        file_path = self._cache_dir / f"{name.replace('.', '_')}.pt"
        torch.save(tensor, file_path)
        self._manifest[name] = file_path
        size = tensor.nelement() * tensor.element_size()
        logger.info(f"NVMe offload: {name} ({size} bytes)")
        return size

    def restore(self, name: str, device: str = "cpu") -> torch.Tensor:
        """Load tensor back from NVMe."""
        if name not in self._manifest:
            raise KeyError(f"Tensor '{name}' not found in NVMe cache")
        tensor = torch.load(self._manifest[name], map_location=device)
        return tensor

    def cleanup(self) -> None:
        """Remove all cached files."""
        for path in self._manifest.values():
            if path.exists():
                path.unlink()
        self._manifest.clear()
        logger.info("NVMe cache cleaned.")

    @property
    def cached_count(self) -> int:
        return len(self._manifest)
