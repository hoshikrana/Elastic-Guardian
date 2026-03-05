"""
EGX NVMe Prober — Layer 2.

Detects disk speeds for planning.
"""

from __future__ import annotations

import os
import time
import tempfile
import pathlib


class NVMeProber:
    """
    Law 1: Storage performance perception.
    """
    
    def probe(self, path: Optional[str] = None) -> Tuple[float, float, int]:
        """Returns (read_gbps, write_gbps, total_bytes)."""
        target = pathlib.Path(path or tempfile.gettempdir())
        
        # Total bytes
        usage = os.statvfs(target) if hasattr(os, 'statvfs') else type('obj', (), {'f_blocks': 0, 'f_frsize': 4096})()
        if hasattr(os, 'statvfs'):
            total_bytes = usage.f_blocks * usage.f_frsize
        else:
            total_bytes = 500 * 1024 * 1024 * 1024 # Windows fallback placeholder
            
        # Benchmarking
        file_size = 256 * 1024 * 1024 # 256MB
        temp_data = os.urandom(file_size)
        temp_file = target / ".egx_probe_tmp"
        
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
        
        os.remove(temp_file)
        
        return float(read_gbps), float(write_gbps), int(total_bytes)
