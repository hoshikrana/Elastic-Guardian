"""
EGX Checkpoint Reader — Layer 4.

Corruption detection + SHA256 verification.
"""

from __future__ import annotations

import hashlib
import torch
import logging
import pathlib
from typing import Any, Dict
from egx.core.exceptions import CheckpointCorruptError


logger = logging.getLogger("egx.resilience.checkpoint")

class CheckpointReader:
    """
    Law 1: Atomic checkpointing.
    Verify SHA256 before any weights enter VRAM.
    """
    
    def load(self, path: str) -> Dict[str, Any]:
        path_obj = pathlib.Path(path)
        sha_path = path_obj.with_suffix(".sha256")
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
            
        # 1. Verify Checksum
        if sha_path.exists():
            expected_sha = sha_path.read_text().strip()
            actual_sha = self._compute_sha256(path_obj)
            
            if expected_sha != actual_sha:
                logger.error(f"CORRUPTION DETECTED: {path}")
                raise CheckpointCorruptError(path=path)
        else:
            logger.warning(f"No SHA256 sidecar found for {path}. Skipping verification.")
            
        # 2. Load
        try:
            return torch.load(path, map_location="cpu")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise CheckpointCorruptError(path=path)

    def _compute_sha256(self, file_path: pathlib.Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
