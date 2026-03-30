"""
EGX Memory Profiler — Layer 4.

Empirical memory profiling during training to validate memory estimates.

Law 2: Each profiler is an isolated instance with no shared state.
Provides real-time GPU/CPU memory monitoring and comparison against estimates.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import deque

try:
    import torch
    import psutil
except ImportError:
    torch = None
    psutil = None

logger = logging.getLogger("egx.monitoring.memory_profiler")


@dataclass
class MemorySnapshot:
    """A single point-in-time memory measurement."""

    timestamp: float
    step: int

    # GPU Memory (bytes)
    gpu_allocated_bytes: int
    gpu_reserved_bytes: int
    gpu_cached_bytes: int

    # CPU Memory (bytes)
    cpu_rss_bytes: int
    cpu_vms_bytes: int

    # Optional metadata
    batch_size: Optional[int] = None
    model_name: Optional[str] = None

    @property
    def gpu_allocated_mb(self) -> float:
        """GPU allocated memory in MB."""
        return self.gpu_allocated_bytes / (1024**2)

    @property
    def gpu_reserved_mb(self) -> float:
        """GPU reserved memory in MB."""
        return self.gpu_reserved_bytes / (1024**2)

    @property
    def cpu_rss_mb(self) -> float:
        """CPU resident set size in MB."""
        return self.cpu_rss_bytes / (1024**2)


@dataclass
class MemoryEstimate:
    """Memory estimate for a model/batch configuration."""

    model_name: str
    batch_size: int
    estimated_gpu_mb: float
    estimated_cpu_mb: float
    training_mode: str
    lora_rank: Optional[int] = None


@dataclass
class ProfilingReport:
    """Summary of profiling results."""

    total_steps: int
    duration_seconds: float

    # Actual measurements
    peak_gpu_allocated_mb: float
    peak_gpu_reserved_mb: float
    peak_cpu_rss_mb: float

    avg_gpu_allocated_mb: float
    avg_gpu_reserved_mb: float
    avg_cpu_rss_mb: float

    # Estimates (if provided)
    estimated_gpu_mb: Optional[float] = None
    estimated_cpu_mb: Optional[float] = None

    # Accuracy metrics
    gpu_accuracy_percent: Optional[float] = None
    cpu_accuracy_percent: Optional[float] = None

    @property
    def accuracy_satisfactory(self) -> bool:
        """Check if accuracy is within ±9% tolerance."""
        if self.gpu_accuracy_percent is None:
            return True
        return abs(100 - self.gpu_accuracy_percent) <= 9.0


class MemoryProfiler:
    """
    Track empirical memory usage during training.

    Samples GPU and CPU memory at regular intervals, compares against
    estimates, and produces profiling reports.

    Thread-safe with no shared state (Law 2).
    """

    def __init__(
        self,
        device_id: int = 0,
        sample_interval: float = 0.1,
        window_size: int = 1000,
    ):
        """
        Initialize memory profiler.

        Args:
            device_id: GPU device ID to monitor (default: 0)
            sample_interval: Time between samples in seconds
            window_size: Rolling window for averaging
        """
        if torch is None or psutil is None:
            raise ImportError("MemoryProfiler requires 'torch' and 'psutil'")

        self.device_id = device_id
        self.sample_interval = sample_interval
        self.window_size = window_size

        self.snapshots: List[MemorySnapshot] = []
        self.estimate: Optional[MemoryEstimate] = None

        self._start_time: Optional[float] = None
        self._process = psutil.Process()
        self._is_profiling = False

        logger.debug(
            f"MemoryProfiler initialized (device={device_id}, "
            f"interval={sample_interval}s, window={window_size})"
        )

    def set_estimate(
        self,
        model_name: str,
        batch_size: int,
        estimated_gpu_mb: float,
        estimated_cpu_mb: float,
        training_mode: str = "full",
        lora_rank: Optional[int] = None,
    ) -> None:
        """
        Set the expected memory estimate for comparison.

        Args:
            model_name: Name of the model
            batch_size: Batch size
            estimated_gpu_mb: Expected GPU memory in MB
            estimated_cpu_mb: Expected CPU memory in MB
            training_mode: Training mode (full, lora, qlora, etc.)
            lora_rank: LoRA rank (if applicable)
        """
        self.estimate = MemoryEstimate(
            model_name=model_name,
            batch_size=batch_size,
            estimated_gpu_mb=estimated_gpu_mb,
            estimated_cpu_mb=estimated_cpu_mb,
            training_mode=training_mode,
            lora_rank=lora_rank,
        )
        logger.info(
            f"Memory estimate set: GPU={estimated_gpu_mb:.1f}MB, "
            f"CPU={estimated_cpu_mb:.1f}MB"
        )

    def start(self) -> None:
        """Start profiling."""
        if self._is_profiling:
            logger.warning("Profiler is already running")
            return

        self.snapshots.clear()
        self._start_time = time.time()
        self._is_profiling = True
        logger.info("Memory profiling started")

    def sample(self, step: int, batch_size: Optional[int] = None) -> MemorySnapshot:
        """
        Take a memory snapshot.

        Args:
            step: Current training step
            batch_size: Current batch size (optional)

        Returns:
            MemorySnapshot with current measurements
        """
        if not self._is_profiling:
            logger.warning("Profiler not running, but sample() called")

        timestamp = time.time()

        # GPU Memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)
            gpu_allocated = torch.cuda.memory_allocated(self.device_id)
            gpu_reserved = torch.cuda.memory_reserved(self.device_id)
            gpu_cached = torch.cuda.memory_reserved(self.device_id) - gpu_allocated
        else:
            gpu_allocated = 0
            gpu_reserved = 0
            gpu_cached = 0

        # CPU Memory
        mem_info = self._process.memory_info()
        cpu_rss = mem_info.rss
        cpu_vms = mem_info.vms

        snapshot = MemorySnapshot(
            timestamp=timestamp,
            step=step,
            gpu_allocated_bytes=gpu_allocated,
            gpu_reserved_bytes=gpu_reserved,
            gpu_cached_bytes=gpu_cached,
            cpu_rss_bytes=cpu_rss,
            cpu_vms_bytes=cpu_vms,
            batch_size=batch_size,
            model_name=self.estimate.model_name if self.estimate else None,
        )

        self.snapshots.append(snapshot)
        return snapshot

    def stop(self) -> ProfilingReport:
        """
        Stop profiling and generate report.

        Returns:
            ProfilingReport with aggregated statistics
        """
        if not self._is_profiling:
            logger.warning("Profiler not running")
            return None

        self._is_profiling = False

        if not self.snapshots:
            logger.warning("No samples collected")
            return None

        duration = time.time() - self._start_time
        total_steps = len(self.snapshots)

        # Extract measurements
        gpu_allocated = [s.gpu_allocated_mb for s in self.snapshots]
        gpu_reserved = [s.gpu_reserved_mb for s in self.snapshots]
        cpu_rss = [s.cpu_rss_mb for s in self.snapshots]

        # Calculate statistics
        peak_gpu_allocated = max(gpu_allocated)
        peak_gpu_reserved = max(gpu_reserved)
        peak_cpu_rss = max(cpu_rss)

        avg_gpu_allocated = sum(gpu_allocated) / len(gpu_allocated)
        avg_gpu_reserved = sum(gpu_reserved) / len(gpu_reserved)
        avg_cpu_rss = sum(cpu_rss) / len(cpu_rss)

        # Calculate accuracy (if estimate provided)
        gpu_accuracy = None
        cpu_accuracy = None

        if self.estimate:
            # Compare peak GPU against estimate
            gpu_accuracy = (self.estimate.estimated_gpu_mb / peak_gpu_allocated) * 100.0
            cpu_accuracy = (self.estimate.estimated_cpu_mb / peak_cpu_rss) * 100.0

        report = ProfilingReport(
            total_steps=total_steps,
            duration_seconds=duration,
            peak_gpu_allocated_mb=peak_gpu_allocated,
            peak_gpu_reserved_mb=peak_gpu_reserved,
            peak_cpu_rss_mb=peak_cpu_rss,
            avg_gpu_allocated_mb=avg_gpu_allocated,
            avg_gpu_reserved_mb=avg_gpu_reserved,
            avg_cpu_rss_mb=avg_cpu_rss,
            estimated_gpu_mb=self.estimate.estimated_gpu_mb if self.estimate else None,
            estimated_cpu_mb=self.estimate.estimated_cpu_mb if self.estimate else None,
            gpu_accuracy_percent=gpu_accuracy,
            cpu_accuracy_percent=cpu_accuracy,
        )

        logger.info("Memory profiling complete")
        logger.info(
            f"Peak GPU: {peak_gpu_allocated:.1f}MB, " f"Peak CPU: {peak_cpu_rss:.1f}MB"
        )
        if gpu_accuracy is not None:
            logger.info(
                f"Estimate accuracy: GPU={gpu_accuracy:.1f}%, CPU={cpu_accuracy:.1f}%"
            )

        return report

    def get_snapshots_window(
        self, window_size: Optional[int] = None
    ) -> List[MemorySnapshot]:
        """
        Get the most recent snapshots (rolling window).

        Args:
            window_size: Number of recent snapshots (default: self.window_size)

        Returns:
            List of MemorySnapshot objects
        """
        if window_size is None:
            window_size = self.window_size

        if len(self.snapshots) <= window_size:
            return self.snapshots[:]

        return self.snapshots[-window_size:]

    def export_csv(self, filepath: str) -> None:
        """
        Export snapshots to CSV file.

        Args:
            filepath: Output file path
        """
        import csv

        if not self.snapshots:
            logger.warning("No snapshots to export")
            return

        try:
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)

                # Header
                header = [
                    "step",
                    "timestamp",
                    "duration_s",
                    "gpu_allocated_mb",
                    "gpu_reserved_mb",
                    "gpu_cached_mb",
                    "cpu_rss_mb",
                    "cpu_vms_mb",
                    "batch_size",
                    "model_name",
                ]
                writer.writerow(header)

                # Data rows
                start_time = self.snapshots[0].timestamp
                for snap in self.snapshots:
                    row = [
                        snap.step,
                        f"{snap.timestamp:.3f}",
                        f"{snap.timestamp - start_time:.3f}",
                        f"{snap.gpu_allocated_mb:.1f}",
                        f"{snap.gpu_reserved_mb:.1f}",
                        f"{snap.gpu_cached_bytes / (1024 ** 2):.1f}",
                        f"{snap.cpu_rss_mb:.1f}",
                        f"{snap.cpu_vms_bytes / (1024 ** 2):.1f}",
                        snap.batch_size or "",
                        snap.model_name or "",
                    ]
                    writer.writerow(row)

            logger.info(f"Profiling data exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")

    def __repr__(self) -> str:
        status = "running" if self._is_profiling else "stopped"
        samples = len(self.snapshots)
        return (
            f"MemoryProfiler(device={self.device_id}, status={status}, "
            f"samples={samples})"
        )
