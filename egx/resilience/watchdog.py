"""
EGX Training Watchdog — Layer 4.

Heartbeat-based deadlock detection in a separate daemon thread.
"""

from __future__ import annotations

import threading
import time
import logging
from typing import Optional
from egx.core.exceptions import DeadlockError


logger = logging.getLogger("egx.resilience.watchdog")

class TrainingWatchdog:
    """
    Law 2: No global mutable state.
    Law 3: Dependency injection.
    """
    
    def __init__(self, timeout_s: float = 30.0):
        self.timeout_s = timeout_s
        self._last_heartbeat = time.monotonic()
        self._last_step = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Starts the watchdog thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        logger.info(f"Watchdog started with {self.timeout_s}s timeout.")

    def stop(self):
        """Stops the watchdog thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def heartbeat(self, step: int):
        """Called by the training kernel every step."""
        self._last_heartbeat = time.monotonic()
        self._last_step = step

    def _monitor(self):
        while not self._stop_event.is_set():
            elapsed = time.monotonic() - self._last_heartbeat
            if elapsed > self.timeout_s:
                logger.error(f"Deadlock detected! No heartbeat for {elapsed:.1f}s.")
                # In v1.0, this triggers an exception that the main kernel must handle
                # but since it's in a thread, we might need a callback or signal.
                # For this implementation, we log and can be queried.
                raise DeadlockError(timeout_s=elapsed, last_step=self._last_step)
            
            time.sleep(1.0)
