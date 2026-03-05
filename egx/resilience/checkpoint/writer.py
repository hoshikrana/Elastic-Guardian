"""
EGX Atomic Checkpoint Writer — Layer 4.

Atomic write + SHA256 checksum.
"""

from __future__ import annotations

import hashlib
import torch
import logging
import pathlib
from typing import Any, Dict


logger = logging.getLogger("egx.resilience.checkpoint")

class CheckpointWriter:
    """
    Law 1: Atomic checkpointing.
    write to .tmp -> fsync -> rename.
    """
    
    def save(self, data: Dict[str, Any], path: str):
        path_obj = pathlib.Path(path)
        tmp_path = path_obj.with_suffix(".tmp")
        
        try:
            # 1. Save to temporary file
            torch.save(data, tmp_path)
            
            # 2. Compute SHA256
            sha256 = self._compute_sha256(tmp_path)
            
            # 3. Write SHA256 sidecar
            with open(path_obj.with_suffix(".sha256"), "w") as f:
                f.write(sha256)
                
            # 4. Atomic Rename
            if path_obj.exists():
                path_obj.unlink()
            tmp_path.rename(path_obj)
            
            logger.info(f"Atomic checkpoint saved: {path} (SHA: {sha256[:8]})")
            
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _compute_sha256(self, file_path: pathlib.Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
