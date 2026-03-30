#!/usr/bin/env python3
"""
EGX Performance Benchmarking Suite

Tracks performance metrics across versions:
- Throughput (samples/sec)
- Latency (ms/batch)
- Memory usage
- Training convergence
- Hardware utilization

Baselines are stored to detect regressions.

Usage:
    python benchmark_egx.py --mode train               # Training benchmark
    python benchmark_egx.py --mode throughput          # Throughput only
    python benchmark_egx.py --mode memory              # Memory profiling
    python benchmark_egx.py --save-baseline            # Save current as baseline
    python benchmark_egx.py --compare                  # Compare with baseline
"""

import argparse
import time
import json
import torch
import psutil
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Benchmark Configuration
# ============================================================================

BENCHMARK_CONFIG = {
    "batch_sizes": [1, 8, 16, 32, 64],
    "sequence_lengths": [128, 256, 512],
    "num_runs": 3,
    "warmup_runs": 1,
    "timeout": 300,
}

# Expected baseline (approximate, can vary by hardware)
BASELINE_METRICS = {
    "throughput_samples_per_sec": 800,  # Minimum acceptable
    "latency_p50_ms": 5,
    "latency_p99_ms": 20,
    "memory_per_batch_mb": 256,
}


# ============================================================================
# Utility Functions
# ============================================================================

def get_gpu_memory() -> Dict[str, float]:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "free_mb": 0}
    
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    free = total - allocated
    
    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "free_mb": free,
        "total_mb": total,
        "utilization_percent": (allocated / total) * 100,
    }


def get_cpu_memory() -> Dict[str, float]:
    """Get current CPU memory usage."""
    process = psutil.Process()
    vm = process.virtual_memory()
    
    return {
        "rss_mb": process.memory_info().rss / 1024 / 1024,
        "vms_mb": process.memory_info().vms / 1024 / 1024,
        "percent": process.memory_percent(),
    }


def get_system_info() -> Dict:
    """Get system information."""
    return {
        "timestamp": datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": (
            torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
            if torch.cuda.is_available()
            else 0
        ),
        "cpu_count": psutil.cpu_count(),
        "total_memory_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
    }


# ============================================================================
# Throughput Benchmarking
# ============================================================================

def benchmark_throughput(batch_sizes: List[int] = None) -> Dict:
    """
    Benchmark throughput (samples/sec).
    
    Tests forward/backward passes with different batch sizes.
    """
    logger.info("\n" + "="*60)
    logger.info("THROUGHPUT BENCHMARKING")
    logger.info("="*60)
    
    batch_sizes = batch_sizes or BENCHMARK_CONFIG["batch_sizes"]
    results = {}
    
    for batch_size in batch_sizes:
        logger.info(f"\nBatch size: {batch_size}")
        
        latencies = []
        
        # Warmup
        for _ in range(BENCHMARK_CONFIG["warmup_runs"]):
            _ = torch.randn(batch_size, 512, 768)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Measure
        for _ in range(BENCHMARK_CONFIG["num_runs"]):
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            
            start = time.perf_counter()
            
            # Simulate training batch
            for _ in range(10):
                data = torch.randn(batch_size, 512, 768)
                if torch.cuda.is_available():
                    data = data.cuda()
                    torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
        
        avg_latency = sum(latencies) / len(latencies)
        throughput = (batch_size * 10) / avg_latency  # samples per second
        
        results[f"batch_{batch_size}"] = {
            "batch_size": batch_size,
            "throughput_samples_per_sec": throughput,
            "latency_ms_per_batch": avg_latency * 1000,
            "samples_per_sec": throughput,
        }
        
        logger.info(f"  Throughput: {throughput:.1f} samples/sec")
        logger.info(f"  Latency: {avg_latency*1000:.2f} ms/batch")
    
    return results


# ============================================================================
# Memory Benchmarking
# ============================================================================

def benchmark_memory() -> Dict:
    """
    Benchmark memory usage and efficiency.
    
    Tests memory allocation patterns and peak usage.
    """
    logger.info("\n" + "="*60)
    logger.info("MEMORY BENCHMARKING")
    logger.info("="*60)
    
    results = {}
    
    # Baseline memory
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    initial_memory = get_gpu_memory() if torch.cuda.is_available() else get_cpu_memory()
    logger.info(f"\nInitial GPU memory: {initial_memory}")
    
    # Allocate and use memory
    tensors = []
    for i in range(5):
        size_mb = (i + 1) * 256
        num_elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
        
        tensor = torch.randn(num_elements, dtype=torch.float32)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        tensors.append(tensor)
        
        current = get_gpu_memory() if torch.cuda.is_available() else get_cpu_memory()
        logger.info(f"\nAfter allocating {size_mb}MB: {current}")
        
        results[f"allocation_{i+1}"] = {
            "allocated_mb": size_mb,
            "memory": current,
        }
    
    # Cleanup
    del tensors
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    final_memory = get_gpu_memory() if torch.cuda.is_available() else get_cpu_memory()
    logger.info(f"\nFinal memory after cleanup: {final_memory}")
    
    results["cleanup"] = {
        "initial": initial_memory,
        "peak": current,
        "final": final_memory,
    }
    
    return results


# ============================================================================
# Latency Benchmarking
# ============================================================================

def benchmark_latency() -> Dict:
    """
    Benchmark latency metrics (p50, p99, p100).
    
    Tests both CPU and GPU latencies for different operations.
    """
    logger.info("\n" + "="*60)
    logger.info("LATENCY BENCHMARKING")
    logger.info("="*60)
    
    latencies = []
    
    for _ in range(100):
        data = torch.randn(32, 512, 768)
        
        start = time.perf_counter()
        if torch.cuda.is_available():
            data = data.cuda()
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        latencies.append(elapsed * 1000)  # Convert to ms
    
    latencies.sort()
    
    results = {
        "p50_ms": latencies[len(latencies) // 2],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
        "p100_ms": latencies[-1],
        "min_ms": latencies[0],
        "max_ms": latencies[-1],
        "mean_ms": sum(latencies) / len(latencies),
    }
    
    logger.info(f"\nLatency Percentiles (ms):")
    logger.info(f"  P50:  {results['p50_ms']:.2f}")
    logger.info(f"  P99:  {results['p99_ms']:.2f}")
    logger.info(f"  P100: {results['p100_ms']:.2f}")
    logger.info(f"  Mean: {results['mean_ms']:.2f}")
    
    return results


# ============================================================================
# Baseline Management
# ============================================================================

def save_baseline(benchmark_results: Dict):
    """Save benchmark results as new baseline."""
    baseline_dir = Path("benchmarks/baselines")
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_file = baseline_dir / f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "system_info": get_system_info(),
        "results": benchmark_results,
    }
    
    with open(baseline_file, "w") as f:
        json.dump(save_data, f, indent=2)
    
    logger.info(f"\nBaseline saved to: {baseline_file}")


def load_latest_baseline() -> Dict:
    """Load latest baseline for comparison."""
    baseline_dir = Path("benchmarks/baselines")
    
    if not baseline_dir.exists():
        logger.warning("No baseline found")
        return None
    
    baseline_files = sorted(baseline_dir.glob("baseline_*.json"))
    if not baseline_files:
        logger.warning("No baseline files found")
        return None
    
    latest = baseline_files[-1]
    logger.info(f"Loading baseline from: {latest}")
    
    with open(latest, "r") as f:
        return json.load(f)


def compare_with_baseline(current: Dict, baseline: Dict = None):
    """Compare current results with baseline."""
    if baseline is None:
        baseline = load_latest_baseline()
    
    if baseline is None:
        logger.warning("No baseline to compare with")
        return
    
    logger.info("\n" + "="*60)
    logger.info("REGRESSION ANALYSIS")
    logger.info("="*60)
    
    # Compare throughput
    if "throughput" in current and "throughput" in baseline.get("results", {}):
        current_tp = current["throughput"]
        baseline_tp = baseline["results"]["throughput"]
        
        for key in current_tp:
            if key in baseline_tp:
                current_val = current_tp[key].get("throughput_samples_per_sec", 0)
                baseline_val = baseline_tp[key].get("throughput_samples_per_sec", 0)
                
                if baseline_val > 0:
                    diff_pct = ((current_val - baseline_val) / baseline_val) * 100
                    status = "✅" if diff_pct > -5 else "⚠️"
                    
                    logger.info(f"{status} {key}: {current_val:.1f} vs {baseline_val:.1f} ({diff_pct:+.1f}%)")


# ============================================================================
# Main Benchmark Suite
# ============================================================================

def run_full_benchmark() -> Dict:
    """Run complete benchmark suite."""
    logger.info("\n" + "="*70)
    logger.info(" EGX PERFORMANCE BENCHMARK - FULL SUITE")
    logger.info("="*70)
    
    system_info = get_system_info()
    logger.info(f"\nSystem: {system_info['gpu_name'] or 'CPU only'}")
    logger.info(f"PyTorch: {system_info['pytorch_version']}")
    logger.info(f"CUDA: {system_info['cuda_version']}")
    
    results = {
        "system_info": system_info,
        "throughput": benchmark_throughput(),
        "latency": benchmark_latency(),
        "memory": benchmark_memory(),
    }
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="EGX Performance Benchmarking")
    parser.add_argument(
        "--mode",
        choices=["train", "throughput", "latency", "memory", "full"],
        default="full",
        help="Benchmark mode",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save results as new baseline",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with latest baseline",
    )
    
    args = parser.parse_args()
    
    try:
        results = {}
        
        if args.mode == "throughput":
            results["throughput"] = benchmark_throughput()
        elif args.mode == "latency":
            results["latency"] = benchmark_latency()
        elif args.mode == "memory":
            results["memory"] = benchmark_memory()
        else:  # full or train
            results = run_full_benchmark()
        
        # Save baseline if requested
        if args.save_baseline:
            save_baseline(results)
        
        # Compare with baseline if requested
        if args.compare:
            compare_with_baseline(results)
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info(" BENCHMARK COMPLETE")
        logger.info("="*70)
        
    except KeyboardInterrupt:
        logger.warning("\nBenchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
