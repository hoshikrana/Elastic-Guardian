"""
EGX Torch Runner — Layer 2.

Isolated dry-run executor.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from typing import Any, Callable
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger("egx.infrastructure.runner")

class TorchRunner:
    """
    Law 3: Executes isolated measurements.
    """
    
    def __init__(self, timeout_s: float = 45.0):
        self.timeout_s = timeout_s
        self.executor = ThreadPoolExecutor(max_workers=1)

    def run_isolated_step(self, model_fn: Callable[[], nn.Module], input_fn: Callable[[], Any]):
        """Runs a training step in a clean environment to measure peak VRAM."""
        future = self.executor.submit(self._execute, model_fn, input_fn)
        try:
            return future.result(timeout=self.timeout_s)
        except Exception as e:
            logger.error(f"Isolated run failed: {e}")
            raise

    def _execute(self, model_fn: Callable[[], nn.Module], input_fn: Callable[[], Any]):
        # Clear cache before
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        model = model_fn()
        inputs = input_fn()
        
        # Warmup
        for _ in range(2):
            output = model(**inputs)
            loss = output.loss if hasattr(output, "loss") else output.sum()
            loss.backward()
            
        # Measurement
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        output = model(**inputs)
        loss = output.loss if hasattr(output, "loss") else output.sum()
        loss.backward()
        
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        
        # Cleanup
        del model, inputs, output, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return peak
