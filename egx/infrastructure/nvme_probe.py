"""
EGX NVMe Prober — Layer 2.

Detects disk speeds for planning.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import tempfile
import time
from typing import Optional, Tuple


class NVMeProber:
    """Probes local storage performance for Level 2/3 caching."""

    __slots__ = ("_temp_path",)

    def __init__(self, temp_path: str = "./egx_nvme_probe.tmp"):
        self._temp_path = temp_path

    def __enter__(self) -> NVMeProber:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if os.path.exists(self._temp_path):
            try:
                os.remove(self._temp_path)
            except Exception:
                pass

    def probe(self, path: Optional[str] = None) -> Tuple[float, float, int]:
        """Returns (read_gbps, write_gbps, total_bytes)."""
        target = pathlib.Path(path or tempfile.gettempdir())

        # Total bytes
        try:
            total, used, free = shutil.disk_usage(target)
            total_bytes = total
        except Exception:
            # Fallback to a sensible default if permission is denied or drive is missing
            total_bytes = 500 * 1024 * 1024 * 1024  # 500GB fallback

        # Benchmarking
        file_size = 256 * 1024 * 1024  # 256MB
        temp_data = os.urandom(file_size)
        temp_file = target / ".egx_probe_tmp"

        try:
            # Write
            start = time.perf_counter()
            with open(temp_file, "wb") as f:
                f.write(temp_data)
                os.fsync(f.fileno())
            write_time = time.perf_counter() - start
            write_gbps = (file_size / write_time) / 1e9

            # Read
            start = time.perf_counter()
            with open(temp_file, "rb") as f:
                _ = f.read()
            read_time = time.perf_counter() - start
            read_gbps = (file_size / read_time) / 1e9
        finally:
            if temp_file.exists():
                os.remove(temp_file)

        return float(read_gbps), float(write_gbps), int(total_bytes)
