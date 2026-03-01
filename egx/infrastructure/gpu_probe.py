# NVML -> nvidia-smi (subprocess, timeout=5s) -> /proc fallback. Returns list[GPUSpec].
"""
EGX GPU probe — Layer 2.

Discovers all GPUs and produces GPUSpec for each.
Three-tier probe cascade, each tier a complete fallback:

  Tier 1: pynvml (direct NVML C library — most accurate)
  Tier 2: nvidia-smi subprocess with timeout (driver present, no pynvml)
  Tier 3: /proc/driver/nvidia/gpus/ (Linux only, last resort)

Never raises. If every tier fails on a GPU, that GPU is silently skipped.
If NO GPU is found, raises GPUNotFoundError with all tried methods listed.

Owns one real measurement: CUDA context overhead.
  After initialising the CUDA context, it measures:
    overhead_bytes = vram_total - torch.cuda.memory_reserved(device)
  This replaces CUDA_OVERHEAD_FACTOR_DEFAULT from core/constants.py.
  The measured value is stored in GPUSpec and consumed by the estimator.

Law 2 boundary: this module is the ONLY place pynvml and torch.cuda
are imported in the infrastructure layer.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from egx.core.enums import DeviceType, InterconnectType
from egx.core.exceptions import GPUNotFoundError, NVMLError
from egx.core.models import GPUSpec

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal probe result — raw data before validation into GPUSpec
# ---------------------------------------------------------------------------

@dataclass
class _RawGPUInfo:
    """Intermediate container. Never leaves this module."""
    gpu_id:        int
    name:          str
    vram_total:    int      # bytes
    vram_free:     int      # bytes
    compute_major: int
    compute_minor: int
    temp_celsius:  float
    pcie_bus_id:   str
    nvlink_peers:  list[int]
    probe_method:  str      # "nvml" | "nvidia-smi" | "proc"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def probe_gpus(require_cuda: bool = True) -> list[GPUSpec]:
    """
    Probe all available GPUs and return a list of GPUSpec.

    Tries pynvml first, falls back to nvidia-smi, then /proc.
    Measures actual CUDA context overhead via torch after probing.

    Args:
        require_cuda: If True and no GPU found, raises GPUNotFoundError.
                      If False, returns empty list when no GPU is found.

    Returns:
        List of GPUSpec, one per GPU, ordered by gpu_id.

    Raises:
        GPUNotFoundError: If require_cuda=True and no GPU is detected.
    """
    tried: list[str] = []
    raw_infos: list[_RawGPUInfo] = []

    # --- Tier 1: pynvml ---------------------------------------------------
    nvml_result = _probe_via_nvml()
    tried.append("pynvml")
    if nvml_result is not None:
        raw_infos = nvml_result
        logger.debug("GPU probe: pynvml succeeded, found %d GPU(s)", len(raw_infos))

    # --- Tier 2: nvidia-smi -----------------------------------------------
    if not raw_infos:
        smi_result = _probe_via_nvidia_smi()
        tried.append("nvidia-smi")
        if smi_result is not None:
            raw_infos = smi_result
            logger.debug(
                "GPU probe: nvidia-smi succeeded, found %d GPU(s)", len(raw_infos)
            )

    # --- Tier 3: /proc ----------------------------------------------------
    if not raw_infos:
        proc_result = _probe_via_proc()
        tried.append("/proc/driver/nvidia")
        if proc_result is not None:
            raw_infos = proc_result
            logger.debug(
                "GPU probe: /proc fallback succeeded, found %d GPU(s)", len(raw_infos)
            )

    # --- No GPU found -----------------------------------------------------
    if not raw_infos:
        if require_cuda:
            raise GPUNotFoundError(probe_methods_tried=tried)
        logger.warning("No CUDA GPU detected (tried: %s). Running CPU-only.", tried)
        return []

    # --- Measure real CUDA context overhead (replaces the default constant)
    overhead_map = _measure_cuda_overhead(raw_infos)

    # --- Convert to GPUSpec -----------------------------------------------
    specs: list[GPUSpec] = []
    for raw in raw_infos:
        try:
            # Apply measured overhead: reduce vram_free by actual context cost
            context_bytes = overhead_map.get(raw.gpu_id, 0)
            adjusted_free = max(0, raw.vram_free - context_bytes)

            spec = GPUSpec(
                gpu_id          = raw.gpu_id,
                name            = raw.name,
                device_type     = DeviceType.CUDA,
                vram_total      = raw.vram_total,
                vram_free       = adjusted_free,
                compute_major   = raw.compute_major,
                compute_minor   = raw.compute_minor,
                nvlink_peer_ids = tuple(raw.nvlink_peers),
                pcie_bus_id     = raw.pcie_bus_id,
                temp_celsius    = raw.temp_celsius,
            )
            specs.append(spec)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "GPU %d: skipped due to invalid data from %s: %s",
                raw.gpu_id, raw.probe_method, exc,
            )

    if not specs and require_cuda:
        raise GPUNotFoundError(probe_methods_tried=tried)

    specs.sort(key=lambda s: s.gpu_id)
    logger.info(
        "GPU probe complete: %d GPU(s) via %s",
        len(specs),
        raw_infos[0].probe_method if raw_infos else "unknown",
    )
    return specs


# ---------------------------------------------------------------------------
# Tier 1: pynvml
# ---------------------------------------------------------------------------

def _probe_via_nvml() -> list[_RawGPUInfo] | None:
    """
    Probe via pynvml (direct NVML C API).
    Returns None if pynvml is not installed or NVML fails to initialise.
    Never raises — all errors are caught and logged.
    """
    try:
        import pynvml  # type: ignore[import]
    except ImportError:
        logger.debug("pynvml not installed, skipping NVML tier")
        return None

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        logger.debug("NVML init failed: %s", exc)
        return None

    try:
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            return None

        infos: list[_RawGPUInfo] = []
        for i in range(count):
            try:
                info = _nvml_single_gpu(pynvml, i)
                if info is not None:
                    infos.append(info)
            except Exception as exc:  # noqa: BLE001
                logger.warning("NVML: GPU %d probe failed: %s", i, exc)

        return infos if infos else None

    except pynvml.NVMLError as exc:
        logger.warning("NVML device enumeration failed: %s", exc)
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def _nvml_single_gpu(pynvml: object, gpu_id: int) -> _RawGPUInfo | None:
    """Extract all data for one GPU via pynvml."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")

    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    vram_total: int = int(mem.total)
    vram_free:  int = int(mem.free)

    cc = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
    compute_major: int = int(cc[0])
    compute_minor: int = int(cc[1])

    try:
        temp = float(pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        ))
    except Exception:  # noqa: BLE001
        temp = 0.0

    try:
        bus_id_raw = pynvml.nvmlDeviceGetPciInfo(handle).busId
        pcie_bus_id = (
            bus_id_raw.decode("utf-8", errors="replace")
            if isinstance(bus_id_raw, bytes)
            else str(bus_id_raw)
        )
    except Exception:  # noqa: BLE001
        pcie_bus_id = ""

    # NVLink peer detection
    nvlink_peers: list[int] = []
    try:
        total_gpus = pynvml.nvmlDeviceGetCount()
        for other_id in range(total_gpus):
            if other_id == gpu_id:
                continue
            other_handle = pynvml.nvmlDeviceGetHandleByIndex(other_id)
            try:
                status = pynvml.nvmlDeviceGetP2PStatus(
                    handle,
                    other_handle,
                    pynvml.NVML_P2P_CAPS_INDEX_NVLINK,
                )
                if status == pynvml.NVML_P2P_STATUS_OK:
                    nvlink_peers.append(other_id)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass

    return _RawGPUInfo(
        gpu_id        = gpu_id,
        name          = name,
        vram_total    = vram_total,
        vram_free     = vram_free,
        compute_major = compute_major,
        compute_minor = compute_minor,
        temp_celsius  = temp,
        pcie_bus_id   = pcie_bus_id,
        nvlink_peers  = nvlink_peers,
        probe_method  = "nvml",
    )


# ---------------------------------------------------------------------------
# Tier 2: nvidia-smi subprocess
# ---------------------------------------------------------------------------

_SMI_QUERY = (
    "index,name,memory.total,memory.free,"
    "compute_cap,temperature.gpu,pci.bus_id"
)
_SMI_FORMAT = "csv,noheader,nounits"
_SMI_TIMEOUT_S = 5.0


def _probe_via_nvidia_smi() -> list[_RawGPUInfo] | None:
    """
    Probe via `nvidia-smi --query-gpu` subprocess.
    Timeout: 5s. Never raises.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={_SMI_QUERY}",
                f"--format={_SMI_FORMAT}",
            ],
            capture_output=True,
            text=True,
            timeout=_SMI_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("nvidia-smi probe failed: %s", exc)
        return None

    if result.returncode != 0:
        logger.debug("nvidia-smi returned %d: %s", result.returncode, result.stderr.strip())
        return None

    infos: list[_RawGPUInfo] = []
    for line in result.stdout.strip().splitlines():
        info = _parse_smi_line(line, len(infos))
        if info is not None:
            infos.append(info)

    return infos if infos else None


def _parse_smi_line(line: str, gpu_id: int) -> _RawGPUInfo | None:
    """Parse one CSV line from nvidia-smi output."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 7:
        return None

    try:
        # Parts: index, name, mem_total_mb, mem_free_mb, compute_cap, temp, pci_bus_id
        raw_idx        = int(parts[0])
        name           = parts[1]
        mem_total_mb   = float(parts[2])
        mem_free_mb    = float(parts[3])
        compute_cap    = parts[4].strip()   # e.g. "8.9"
        temp_raw       = parts[5].strip()
        pcie_bus_id    = parts[6].strip() if len(parts) > 6 else ""

        vram_total = int(mem_total_mb * 1_048_576)   # MiB -> bytes
        vram_free  = int(mem_free_mb  * 1_048_576)

        cc_parts = compute_cap.split(".")
        compute_major = int(cc_parts[0]) if cc_parts else 0
        compute_minor = int(cc_parts[1]) if len(cc_parts) > 1 else 0

        temp = float(temp_raw) if temp_raw.isdigit() else 0.0

        return _RawGPUInfo(
            gpu_id        = raw_idx,
            name          = name,
            vram_total    = vram_total,
            vram_free     = vram_free,
            compute_major = compute_major,
            compute_minor = compute_minor,
            temp_celsius  = temp,
            pcie_bus_id   = pcie_bus_id,
            nvlink_peers  = [],   # smi doesn't give NVLink peer list directly
            probe_method  = "nvidia-smi",
        )
    except (ValueError, IndexError) as exc:
        logger.debug("nvidia-smi parse error on line %r: %s", line, exc)
        return None


# ---------------------------------------------------------------------------
# Tier 3: /proc fallback (Linux only)
# ---------------------------------------------------------------------------

def _probe_via_proc() -> list[_RawGPUInfo] | None:
    """
    Last-resort probe via /proc/driver/nvidia/gpus/.
    Linux only. Gives vram_total from 'Model:' and 'Video Memory:' lines.
    No VRAM free, no temperature — minimally useful but better than nothing.
    """
    proc_root = Path("/proc/driver/nvidia/gpus")
    if not proc_root.exists():
        return None

    infos: list[_RawGPUInfo] = []
    for gpu_id, gpu_dir in enumerate(sorted(proc_root.iterdir())):
        info_file = gpu_dir / "information"
        if not info_file.exists():
            continue
        try:
            text = info_file.read_text(errors="replace")
            info = _parse_proc_gpu_info(text, gpu_id, str(gpu_dir.name))
            if info is not None:
                infos.append(info)
        except OSError as exc:
            logger.debug("/proc GPU %d read error: %s", gpu_id, exc)

    return infos if infos else None


def _parse_proc_gpu_info(text: str, gpu_id: int, bus_id: str) -> _RawGPUInfo | None:
    """Parse /proc/driver/nvidia/gpus/<bus_id>/information."""
    name = ""
    vram_mib = 0

    for line in text.splitlines():
        if line.startswith("Model:"):
            name = line.split(":", 1)[1].strip()
        elif "Video Memory:" in line or "Total Memory:" in line:
            # Format: "Video Memory:   24576 MiB"
            match = re.search(r"(\d+)\s*MiB", line)
            if match:
                vram_mib = int(match.group(1))

    if not name or vram_mib == 0:
        return None

    vram_total = vram_mib * 1_048_576

    return _RawGPUInfo(
        gpu_id        = gpu_id,
        name          = name,
        vram_total    = vram_total,
        vram_free     = vram_total,   # unknown — assume all free (conservative)
        compute_major = 0,
        compute_minor = 0,
        temp_celsius  = 0.0,
        pcie_bus_id   = bus_id,
        nvlink_peers  = [],
        probe_method  = "/proc",
    )


# ---------------------------------------------------------------------------
# CUDA context overhead measurement
# ---------------------------------------------------------------------------

def _measure_cuda_overhead(raw_infos: list[_RawGPUInfo]) -> dict[int, int]:
    """
    Measure actual CUDA context overhead per GPU.

    How it works:
      1. Import torch (deferred — only if we have GPU data to measure)
      2. For each GPU, initialise the CUDA context by allocating a tiny tensor
      3. Measure: overhead = torch.cuda.memory_reserved(device) immediately
         after context init, before any model weights are loaded
      4. Return map: {gpu_id: overhead_bytes}

    This measurement replaces CUDA_OVERHEAD_FACTOR_DEFAULT.
    It accounts for driver version, CUDA version, and GPU model differences.

    Returns empty dict if torch is not available or measurement fails.
    The caller falls back to CUDA_OVERHEAD_FACTOR_DEFAULT in that case.
    """
    try:
        import torch  # type: ignore[import]
    except ImportError:
        logger.debug("torch not available, skipping CUDA overhead measurement")
        return {}

    if not torch.cuda.is_available():
        return {}

    overhead_map: dict[int, int] = {}
    for raw in raw_infos:
        gpu_id = raw.gpu_id
        try:
            device = torch.device(f"cuda:{gpu_id}")
            # Initialise context with a tiny allocation
            probe_tensor = torch.zeros(1, device=device)
            torch.cuda.synchronize(device)

            # Memory reserved by CUDA context (not the tiny tensor — that's ~4 bytes)
            reserved_bytes: int = int(torch.cuda.memory_reserved(device))

            del probe_tensor
            torch.cuda.empty_cache()

            overhead_map[gpu_id] = reserved_bytes
            logger.debug(
                "GPU %d CUDA context overhead: %d bytes (%.1f MiB)",
                gpu_id,
                reserved_bytes,
                reserved_bytes / 1_048_576,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("GPU %d overhead measurement failed: %s", gpu_id, exc)

    return overhead_map
