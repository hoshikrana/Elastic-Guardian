"""
EGX Core Constants — Layer 1.

No logic. No classes. No functions. Constants only.
"""

from .enums import TrainingMode

# Units (Law 10: all arithmetic uses int bytes)
KB = 1024
MB = 1024 * 1024
GB = 1024 * 1024 * 1024
TB = 1024 * 1024 * 1024 * 1024

# System Constants
WATCHDOG_TIMEOUT_S = 30.0
HEARTBEAT_INTERVAL_S = 1.0

# Safety Thresholds (VRAM fraction per mode)
SAFETY_THRESHOLDS = {
    TrainingMode.FULL_FINETUNE: 0.72,
    TrainingMode.LORA_PLUS: 0.80,
    TrainingMode.LORA: 0.80,
    TrainingMode.DORA: 0.82,
    TrainingMode.QLORA: 0.90,
}

# Strategy Priority Order
STRATEGY_PRIORITY_ORDER = [
    TrainingMode.FULL_FINETUNE,
    TrainingMode.LORA_PLUS,
    TrainingMode.LORA,
    TrainingMode.DORA,
    TrainingMode.QLORA,
]
