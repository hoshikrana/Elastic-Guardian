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
    write to .tmp -> fsync -> os.replace.
    """

    __slots__ = ()

    def save(self, data: Dict[str, Any], path: str):
        path_obj = pathlib.Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path_obj.with_suffix(".tmp")

        try:
            # 1. Save to temporary file
            # If all values are tensors, use safetensors for security
            can_use_safetensors = all(isinstance(v, torch.Tensor) for v in data.values()) if isinstance(data, dict) else False
            
            if can_use_safetensors:
                from safetensors.torch import save_file
                save_file(data, tmp_path)
            else:
                torch.save(data, tmp_path)

            # 2. Compute SHA256
            sha256 = self._compute_sha256(tmp_path)

            # 3. Write SHA256 sidecar (Atomic sidecar too)
            sidecar_path = path_obj.with_suffix(".sha256")
            sidecar_tmp = sidecar_path.with_suffix(".sha256.tmp")
            with open(sidecar_tmp, "w") as f:
                f.write(sha256)
                f.flush()
                import os
                os.fsync(f.fileno())
            
            # 4. Atomic Replace
            os.replace(sidecar_tmp, sidecar_path)
            os.replace(tmp_path, path_obj)

            logger.info("Atomic checkpoint saved: %s (SHA: %s)", path, sha256[:8])

        except Exception as e:
            logger.error("Checkpoint save failed: %s", e)
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _compute_sha256(self, file_path: pathlib.Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
