"""
EGX API Config — Layer 7.

Runtime configuration with sane defaults and override support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, Callable, List

from egx.core.enums import OptimizerType


@dataclass
class EGXConfig:
    """
    Master configuration for an EGX training session.
    All fields have sensible defaults — zero-config is the default.
    """

    # Model
    model_name_or_path: Optional[str] = None
    scratch: bool = False
    scratch_config_path: Optional[str] = None

    # Training
    max_steps: int = -1
    num_epochs: int = 3
    optimizer_type: Union[str, OptimizerType] = "adamw"
    loss_fn: Optional[Union[str, Callable]] = None
    scheduler_type: Optional[str] = None
    warmup_steps: int = 0
    precision_override: Optional[str] = None
    callbacks: List[Callable] = field(default_factory=list)
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # PEFT
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: Optional[tuple] = None

    # Data
    dataset_path: Optional[str] = None
    max_seq_length: int = 2048

    # Checkpointing
    output_dir: str = "./egx_output"
    checkpoint_strategy: str = "adaptive"
    save_total_limit: int = 3

    # Hardware
    device: str = "auto"

    # Advanced
    overrides: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-notation config access with fallback."""
        return self.overrides.get(key, getattr(self, key, default))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EGXConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        init_args = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        cfg = cls(**init_args)
        cfg.overrides.update(extra)
        return cfg
