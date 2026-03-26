"""
EGX Training Watchdog — Layer 4.

Heartbeat-based deadlock detection in a separate daemon thread.
Uses Event + stored exception pattern to propagate errors to the
training loop (exceptions raised in daemon threads are silently
swallowed by Python).
"""

from __future__ import annotations

import threading
import time
import logging
from typing import Optional
from egx.core.exceptions import DeadlockError
from egx.core.interfaces import BaseWatchdog

logger = logging.getLogger("egx.resilience.watchdog")


class TrainingWatchdog(BaseWatchdog):
    """
    Law 2: No global mutable state.
    Law 3: Dependency injection.

    The watchdog runs in a daemon thread and sets an Event when a
    deadlock is detected.  The training kernel must call ``check()``
    each step to propagate the stored error into the main thread.
    """

    def __init__(self, timeout_s: float = 300.0):
        self.timeout_s = timeout_s
        self._last_heartbeat = time.monotonic()
        self._last_step = 0
        self._stop_event = threading.Event()
        self._deadlock_detected = threading.Event()
        self._deadlock_error: Optional[DeadlockError] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Starts the watchdog thread."""
        self._stop_event.clear()
        self._deadlock_detected.clear()
        self._deadlock_error = None
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        logger.info("Watchdog started with %.1fs timeout.", self.timeout_s)

    def stop(self):
        """Stops the watchdog thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def heartbeat(self, step: int):
        """Called by the training kernel every step."""
        self._last_heartbeat = time.monotonic()
        self._last_step = step

    def check(self):
        """Raise the stored deadlock error if one was detected.

        Must be called from the main training thread so the exception
        propagates correctly.
        """
        if self._deadlock_detected.is_set() and self._deadlock_error is not None:
            raise self._deadlock_error

    def _monitor(self):
        while not self._stop_event.is_set():
            elapsed = time.monotonic() - self._last_heartbeat
            if elapsed > self.timeout_s:
                logger.error(
                    "Deadlock detected! No heartbeat for %.1fs.", elapsed,
                )
                self._deadlock_error = DeadlockError(
                    timeout_s=elapsed, last_step=self._last_step,
                )
                self._deadlock_detected.set()
                return  # exit the monitoring loop

            time.sleep(1.0)
