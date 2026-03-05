"""
EGX Telemetry Service — Layer 4.

Broadcaster for training logs to console, disk, and cloud providers.
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("egx.telemetry")


class TelemetryService:
    """
    Handles training observation.
    """
    
    def __init__(self, log_dir: str = "./logs/egx"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = int(time.time())
        self.history_file = self.log_dir / f"training_{self.session_id}.jsonl"

    def broadcast_step(self, step: int, metrics: Dict[str, Any]):
        """Logs a step to the jsonl file and console."""
        payload = {
            "step": step,
            "timestamp": time.time(),
            **metrics
        }
        
        # 1. Write to local storage
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
            
        # 2. Console Summary
        if step % 10 == 0:
            loss = metrics.get("loss", 0.0)
            tps = metrics.get("tokens_per_sec", 0.0)
            logger.info(f"EGX Step {step}: Loss={loss:.4f} | Throughput={tps:.1f} tokens/s")
