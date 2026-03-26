"""
EGX API Exports — Layer 7.
"""

from egx.api.trainer import EGXTrainer as EGX
from egx.api.config import EGXConfig
from egx.api.callbacks import (
    TrainingCallback,
    CallbackHandler,
    EarlyStoppingCallback,
    LoggingCallback,
    GradientClipCallback,
    NaNDetectionCallback,
    ThroughputCallback,
    CheckpointCallback,
)
from egx.api.evaluator import EGXEvaluator
from egx.api.predictor import EGXPredictor

__all__ = [
    "EGX",
    "EGXConfig",
    "EGXTrainer",
    "TrainingCallback",
    "CallbackHandler",
    "EarlyStoppingCallback",
    "LoggingCallback",
    "GradientClipCallback",
    "NaNDetectionCallback",
    "ThroughputCallback",
    "CheckpointCallback",
    "EGXEvaluator",
    "EGXPredictor",
]

# Alias
EGXTrainer = EGX
