"""
EGX Structured Logger — Layer 2.

JSON structured logging, PID-scoped.
"""

from __future__ import annotations

import logging
import json
import os
import time


class StructuredLogger:
    """
    Law 9: No silent fallbacks — log every degradation.
    """

    def __init__(self, name: str = "egx"):
        self.logger = logging.getLogger(name)
        self.pid = os.getpid()

    def log_event(self, event_type: str, data: dict, level: int = logging.INFO):
        payload = {
            "timestamp": time.time(),
            "pid": self.pid,
            "event": event_type,
            "data": data,
        }
        self.logger.log(level, json.dumps(payload))

    def log_degradation(self, component: str, reason: str, action: str):
        self.log_event(
            "degradation",
            {"component": component, "reason": reason, "action": action},
            level=logging.WARNING,
        )

    def log_decision(self, run_id: str, decision: dict):
        self.log_event("training_decision", {"run_id": run_id, "decision": decision})
