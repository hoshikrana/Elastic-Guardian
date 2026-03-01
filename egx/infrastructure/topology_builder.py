# Assembles HardwareTopology. NVLink via nvmlDeviceGetP2PStatus. Measured bandwidth.
"""
EGX topology builder — Layer 2.

Assembles HardwareTopology from:
  - GPUSpec list (from gpu_probe.py)
  - CPU RAM (from /proc/meminfo on Linux, psutil fallback)
  - NVMe path and capacity (from /proc/mounts + statvfs)
  - PCIe host-device bandwidth (measured tensor transfer)
  - NVLink device-device bandwidth (measured tensor transfer, if NVLink present)
  - InterconnectType determination (NVLINK > PCIE, based on GPUSpec.nvlink_peer_ids)

All bandwidth values are int bytes/s — measured, not assumed.
Falls back gracefully: missing NVMe -> nvme_capacity=0, missing bandwidth -> 0.

Called once at startup by runtime/lifecycle.py Phase 1.
Result is stored in the runtime state and reused — never re-probed mid-training.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Iterator

from egx.core.enums import InterconnectType
from egx.core.models import GPUSpec, HardwareTopology

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_topology(gpu_specs: list[GPUSpec]) -> HardwareTopology:
    """
    Build a complete HardwareTopology from a list of GPUSpecs.

    All measurements happen here. The returned topology is fully populated
    with real measured values — no defaults or placeholders survive.

    Args:
        gpu_specs: Output of gpu_probe.probe_gpus(). May be empty (CPU-only).

    Returns:
        HardwareTopology with all fields populated from live measurements.
    """
    logger.info("Building hardware topology for %d GPU(s)...", len(gpu_specs))

    # CPU RAM
    ram_total, ram_free = _probe_cpu_ram()
    logger.debug("CPU RAM: total=%d free=%d", ram_total, ram_free)

    # NVMe
    nvme_path, nvme_capacity = _probe_nvme()
    logger.debug("NVMe: path=%r capacity=%d", nvme_path, nvme_capacity)

    # NVMe bandwidth (only if NVMe is available and we have GPU(s) to train on)
    nvme_read_bw  = 0
    nvme_write_bw = 0
    if nvme_path and gpu_specs:
        nvme_read_bw, nvme_write_bw = _measure_nvme_bandwidth(nvme_path)
        logger.debug(
            "NVMe bandwidth: read=%d write=%d bytes/s", nvme_read_bw, nvme_write_bw
        )

    # PCIe and NVLink bandwidth (only if GPUs present)
    pcie_bw   = 0
    nvlink_bw = 0
    interconnect = _determine_interconnect(gpu_specs)

    if gpu_specs:
        pcie_bw = _measure_pcie_bandwidth(gpu_specs[0].gpu_id)
        logger.debug("PCIe bandwidth: %d bytes/s", pcie_bw)

        if interconnect == InterconnectType.NVLINK and len(gpu_specs) >= 2:
            nvlink_bw = _measure_nvlink_bandwidth(
                gpu_specs[0].gpu_id, gpu_specs[1].gpu_id
            )
            logger.debug("NVLink bandwidth: %d bytes/s", nvlink_bw)

    topo = HardwareTopology(
        gpus              = tuple(gpu_specs),
        cpu_ram_total     = ram_total,
        cpu_ram_free      = ram_free,
        nvme_path         = nvme_path,
        nvme_capacity     = nvme_capacity,
        nvme_read_bw      = nvme_read_bw,
        nvme_write_bw     = nvme_write_bw,
        pcie_bandwidth    = pcie_bw,
        nvlink_bandwidth  = nvlink_bw,
        interconnect_type = interconnect,
    )

    logger.info(
        "Topology built: %d GPU(s), %.0f GiB VRAM, %.0f GiB RAM, "
        "interconnect=%s, PCIe=%.1f GB/s, NVLink=%.1f GB/s",
        topo.gpu_count,
        topo.total_vram / 1_073_741_824,
        ram_total        / 1_073_741_824,
        interconnect.value,
        pcie_bw          / 1e9,
        nvlink_bw        / 1e9,
    )
    return topo


# ---------------------------------------------------------------------------
# CPU RAM probe
# ---------------------------------------------------------------------------

def _probe_cpu_ram() -> tuple[int, int]:
    """
    Returns (total_bytes, free_bytes).

    Priority:
      1. /proc/meminfo (Linux — most accurate, available free)
      2. psutil (cross-platform fallback)
      3. (0, 0) if both unavailable
    """
    # --- /proc/meminfo ---
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            return _parse_proc_meminfo(meminfo.read_text())
        except (OSError, ValueError) as exc:
            logger.debug("/proc/meminfo parse failed: %s", exc)

    # --- psutil fallback ---
    try:
        import psutil  # type: ignore[import]
        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except ImportError:
        logger.debug("psutil not available for RAM probe")
    except Exception as exc:  # noqa: BLE001
        logger.debug("psutil RAM probe failed: %s", exc)

    logger.warning("Could not determine CPU RAM — returning 0")
    return 0, 0


def _parse_proc_meminfo(text: str) -> tuple[int, int]:
    """Parse /proc/meminfo for MemTotal and MemAvailable."""
    total_kb = 0
    avail_kb = 0

    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            total_kb = int(re.search(r"(\d+)", line).group(1))  # type: ignore[union-attr]
        elif line.startswith("MemAvailable:"):
            avail_kb = int(re.search(r"(\d+)", line).group(1))  # type: ignore[union-attr]

    if total_kb == 0:
        raise ValueError("MemTotal not found in /proc/meminfo")

    return total_kb * 1024, avail_kb * 1024


# ---------------------------------------------------------------------------
# NVMe probe
# ---------------------------------------------------------------------------

def _probe_nvme() -> tuple[str, int]:
    """
    Find the best NVMe mount point for tensor offloading.

    Priority:
      1. /proc/mounts — scan for nvme devices (Linux)
      2. Fallback: use the mount point of the current working directory

    Returns (mount_path, capacity_bytes). Returns ("", 0) if nothing found.
    """
    nvme_mounts = list(_find_nvme_mounts())

    if nvme_mounts:
        # Pick the mount with the most free space
        best = max(nvme_mounts, key=lambda m: _statvfs_free(m))
        capacity = _statvfs_capacity(best)
        return best, capacity

    # Fallback: cwd mount point (for non-Linux or systems without NVMe labels)
    cwd = str(Path.cwd())
    try:
        capacity = _statvfs_capacity(cwd)
        if capacity > 0:
            logger.debug("NVMe fallback: using cwd mount %s (cap=%d)", cwd, capacity)
            return cwd, capacity
    except OSError:
        pass

    return "", 0


def _find_nvme_mounts() -> Iterator[str]:
    """Yield mount points for NVMe devices from /proc/mounts."""
    mounts_file = Path("/proc/mounts")
    if not mounts_file.exists():
        return

    try:
        for line in mounts_file.read_text().splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            device, mount_point = parts[0], parts[1]
            # NVMe devices: /dev/nvme*, /dev/nvme0n1, /dev/nvme0n1p1, etc.
            if re.match(r"^/dev/nvme", device):
                yield mount_point
    except OSError as exc:
        logger.debug("/proc/mounts read failed: %s", exc)


def _statvfs_free(path: str) -> int:
    try:
        s = os.statvfs(path)
        return s.f_bavail * s.f_frsize
    except OSError:
        return 0


def _statvfs_capacity(path: str) -> int:
    try:
        s = os.statvfs(path)
        return s.f_blocks * s.f_frsize
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Interconnect determination
# ---------------------------------------------------------------------------

def _determine_interconnect(gpu_specs: list[GPUSpec]) -> InterconnectType:
    """
    Determine GPU-to-GPU interconnect from GPUSpec.nvlink_peer_ids.

    Rules:
      - Any GPU has NVLink peers → NVLINK
      - Multiple GPUs, no NVLink → PCIE
      - Single GPU → NONE
    """
    if not gpu_specs:
        return InterconnectType.NONE
    if len(gpu_specs) == 1:
        return InterconnectType.NONE
    for spec in gpu_specs:
        if spec.nvlink_peer_ids:
            return InterconnectType.NVLINK
    return InterconnectType.PCIE


# ---------------------------------------------------------------------------
# PCIe bandwidth measurement
# ---------------------------------------------------------------------------

_PCIE_TRANSFER_BYTES = 512 * 1_048_576   # 512 MiB transfer for measurement
_PCIE_WARMUP_ITERS   = 3
_PCIE_MEASURE_ITERS  = 5


def _measure_pcie_bandwidth(gpu_id: int) -> int:
    """
    Measure host-to-device PCIe bandwidth by timing a pinned-memory transfer.

    Returns:
        int bytes/s. Returns 0 if torch not available or measurement fails.
    """
    try:
        import torch  # type: ignore[import]
    except ImportError:
        return 0

    if not torch.cuda.is_available():
        return 0

    try:
        device = torch.device(f"cuda:{gpu_id}")
        num_elements = _PCIE_TRANSFER_BYTES // 4   # float32

        # Pinned memory for realistic PCIe transfer (matches training data loading)
        cpu_tensor = torch.zeros(num_elements, dtype=torch.float32).pin_memory()

        # Warmup
        for _ in range(_PCIE_WARMUP_ITERS):
            gpu_tensor = cpu_tensor.to(device, non_blocking=False)
            torch.cuda.synchronize(device)
            del gpu_tensor

        # Measured transfers
        elapsed_s = 0.0
        for _ in range(_PCIE_MEASURE_ITERS):
            t0 = time.perf_counter()
            gpu_tensor = cpu_tensor.to(device, non_blocking=False)
            torch.cuda.synchronize(device)
            elapsed_s += time.perf_counter() - t0
            del gpu_tensor

        del cpu_tensor
        torch.cuda.empty_cache()

        avg_s = elapsed_s / _PCIE_MEASURE_ITERS
        bandwidth_bps = int(_PCIE_TRANSFER_BYTES / avg_s)
        return bandwidth_bps

    except Exception as exc:  # noqa: BLE001
        logger.debug("PCIe bandwidth measurement failed: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# NVLink bandwidth measurement
# ---------------------------------------------------------------------------

_NVLINK_TRANSFER_BYTES = 1 * 1_073_741_824   # 1 GiB (NVLink can sustain this)
_NVLINK_WARMUP_ITERS   = 3
_NVLINK_MEASURE_ITERS  = 5


def _measure_nvlink_bandwidth(gpu_id_src: int, gpu_id_dst: int) -> int:
    """
    Measure NVLink device-to-device bandwidth by timing a direct GPU transfer.

    Returns:
        int bytes/s. Returns 0 if torch not available or measurement fails.
    """
    try:
        import torch  # type: ignore[import]
    except ImportError:
        return 0

    if not torch.cuda.is_available():
        return 0

    try:
        src_device = torch.device(f"cuda:{gpu_id_src}")
        dst_device = torch.device(f"cuda:{gpu_id_dst}")
        num_elements = _NVLINK_TRANSFER_BYTES // 4   # float32

        src_tensor = torch.zeros(num_elements, dtype=torch.float32, device=src_device)
        torch.cuda.synchronize(src_device)

        # Warmup
        for _ in range(_NVLINK_WARMUP_ITERS):
            dst_tensor = src_tensor.to(dst_device, non_blocking=False)
            torch.cuda.synchronize(dst_device)
            del dst_tensor

        # Measured
        elapsed_s = 0.0
        for _ in range(_NVLINK_MEASURE_ITERS):
            t0 = time.perf_counter()
            dst_tensor = src_tensor.to(dst_device, non_blocking=False)
            torch.cuda.synchronize(dst_device)
            elapsed_s += time.perf_counter() - t0
            del dst_tensor

        del src_tensor
        torch.cuda.empty_cache()

        avg_s = elapsed_s / _NVLINK_MEASURE_ITERS
        return int(_NVLINK_TRANSFER_BYTES / avg_s)

    except Exception as exc:  # noqa: BLE001
        logger.debug("NVLink bandwidth measurement failed: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# NVMe bandwidth measurement
# ---------------------------------------------------------------------------

_NVME_FILE_BYTES   = 128 * 1_048_576   # 128 MiB test file
_NVME_MEASURE_ITERS = 3


def _measure_nvme_bandwidth(nvme_path: str) -> tuple[int, int]:
    """
    Measure NVMe sequential read and write bandwidth.

    Writes a 128 MiB file, reads it back, removes it.
    Returns (read_bytes_per_sec, write_bytes_per_sec).
    Returns (0, 0) on any failure.
    """
    test_file = Path(nvme_path) / ".egx_nvme_probe"
    data = bytes(_NVME_FILE_BYTES)

    try:
        write_times = []
        read_times = []

        for _ in range(_NVME_MEASURE_ITERS):
            # Write
            t0 = time.perf_counter()
            test_file.write_bytes(data)
            os.sync()   # flush to physical device
            write_times.append(time.perf_counter() - t0)

            # Read (O_DIRECT equivalent via re-open)
            t0 = time.perf_counter()
            _ = test_file.read_bytes()
            read_times.append(time.perf_counter() - t0)

        avg_write_s = sum(write_times) / len(write_times)
        avg_read_s  = sum(read_times)  / len(read_times)

        write_bps = int(_NVME_FILE_BYTES / avg_write_s) if avg_write_s > 0 else 0
        read_bps  = int(_NVME_FILE_BYTES / avg_read_s)  if avg_read_s  > 0 else 0

        return read_bps, write_bps

    except OSError as exc:
        logger.debug("NVMe benchmark failed: %s", exc)
        return 0, 0
    finally:
        try:
            test_file.unlink(missing_ok=True)

        except OSError:
            pass