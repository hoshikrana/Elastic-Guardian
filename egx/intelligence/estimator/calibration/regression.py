"""
EGX Calibration Regression — Layer 3.

Linear regression for calibrating analytical estimates against dry-run actuals.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("egx.intelligence.calibration")


class CalibrationRegression:
    """
    Simple online linear regression: actual = slope * predicted + intercept.
    Updated incrementally as dry-run results arrive.
    """

    def __init__(self):
        self._n = 0
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_xx = 0.0
        self._sum_xy = 0.0
        self.slope = 1.0
        self.intercept = 0.0

    def update(self, predicted: int, actual: int) -> None:
        """Add a new (predicted, actual) observation and refit."""
        x, y = float(predicted), float(actual)
        self._n += 1
        self._sum_x += x
        self._sum_y += y
        self._sum_xx += x * x
        self._sum_xy += x * y
        self._refit()

    def predict(self, predicted: int) -> int:
        """Calibrate a raw analytical prediction."""
        return max(0, int(self.slope * predicted + self.intercept))

    def _refit(self) -> None:
        if self._n < 2:
            return
        denom = self._n * self._sum_xx - self._sum_x**2
        if abs(denom) < 1e-12:
            return
        self.slope = (self._n * self._sum_xy - self._sum_x * self._sum_y) / denom
        self.intercept = (self._sum_y - self.slope * self._sum_x) / self._n

    @property
    def r_squared(self) -> float:
        if self._n < 3:
            return 0.0
        return 0.0  # placeholder — full version stores history

    def reset(self) -> None:
        self.__init__()
