"""
EGX Hybrid Estimator — Layer 3.

Combines analytical + dry-run estimates with calibration regression.
"""

from __future__ import annotations

import logging

from egx.core.models import GPUSpec
from egx.intelligence.estimator.calibration.regression import CalibrationRegression
from egx.intelligence.estimator.calibration.cache import CalibrationCache

logger = logging.getLogger("egx.intelligence.estimator")


class HybridEstimator:
    """
    Law 11: Analytical O(1) estimate, then calibrate with regression.
    """

    def __init__(self):
        self._regression = CalibrationRegression()
        self._cache = CalibrationCache()

    def estimate_vram(
        self,
        model_params: int,
        bytes_per_param: int,
        gpu: GPUSpec,
        mode: str,
    ) -> int:
        """Returns calibrated VRAM estimate in bytes."""
        cache_key = self._cache.make_key(
            model_name=f"params_{model_params}",
            gpu_name=gpu.name,
            vram_bytes=gpu.vram_bytes,
            mode=mode,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached["vram_bytes"]

        # Analytical estimate
        raw = model_params * bytes_per_param
        # Calibrate
        calibrated = self._regression.predict(raw)
        if calibrated == 0:
            calibrated = raw  # No calibration data yet

        self._cache.put(cache_key, {"vram_bytes": calibrated})
        return calibrated

    def record_actual(self, predicted: int, actual: int) -> None:
        """Feed dry-run actuals back to improve future estimates."""
        self._regression.update(predicted, actual)
        logger.info(
            f"Calibration updated: slope={self._regression.slope:.3f}, "
            f"intercept={self._regression.intercept:.0f}"
        )
