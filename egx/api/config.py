"""
EGX API Config — Layer 7.

Runtime configuration with sane defaults and override support.
Covers training, evaluation, generation, data, checkpointing, and hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, Callable, List

from egx.core.enums import OptimizerType


@dataclass(frozen=True, slots=True)
class EGXConfig:
    """
    Master configuration for an EGX training session.
    All fields have sensible defaults — zero-config is the default.
    """

    # ── Model ──
    model_name_or_path: Optional[str] = None
    scratch: bool = False
    scratch_config_path: Optional[str] = None

    # ── Training ──
    max_steps: int = -1
    num_epochs: int = 3
    batch_size: int = 2
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
    gradient_checkpointing: bool = False

    # ── PEFT ──
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: Optional[tuple] = None

    # ── Data ──
    dataset_path: Optional[str] = None
    max_seq_length: int = 2048

    # ── Evaluation ──
    eval_batch_size: int = 4
    eval_strategy: str = "epoch"        # "epoch", "steps", "no"
    eval_steps: int = 500               # evaluate every N steps (if eval_strategy="steps")
    metric_for_best_model: str = "loss"
    compute_perplexity: bool = True

    # ── Early Stopping ──
    early_stopping_patience: int = 0    # 0 = disabled
    early_stopping_threshold: float = 0.0

    # ── Logging ──
    logging_steps: int = 10
    log_level: str = "info"

    # ── Checkpointing ──
    output_dir: str = "./egx_output"
    checkpoint_strategy: str = "adaptive"
    save_total_limit: int = 3
    save_steps: int = 500
    timeout: float = 300.0              # Law 4: Defensive defaults (5 min for cold starts)

    # ── Hardware ──
    device: str = "auto"

    # ── Advanced ──
    overrides: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-notation config access with fallback to overrides."""
        return self.overrides.get(key, getattr(self, key, default))

    def __post_init__(self):
        """Validate configuration at construction time."""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_epochs < 1 and self.max_steps <= 0:
            raise ValueError(f"num_epochs must be >= 1 (or set max_steps > 0), got {self.num_epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.max_grad_norm < 0:
            raise ValueError(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")
        if self.eval_strategy not in ("epoch", "steps", "no"):
            raise ValueError(
                f"eval_strategy must be 'epoch', 'steps', or 'no', got '{self.eval_strategy}'"
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EGXConfig":
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        init_args = {k: v for k, v in d.items() if k in known and k != "overrides"}
        extra = {k: v for k, v in d.items() if k not in known}
        if "overrides" in d and isinstance(d["overrides"], dict):
            extra.update(d["overrides"])
        return cls(**init_args, overrides=extra)

    def __repr__(self) -> str:
        return (
            f"EGXConfig(lr={self.learning_rate}, epochs={self.num_epochs}, "
            f"batch={self.batch_size}, eval='{self.eval_strategy}')"
        )
