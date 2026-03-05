"""
EGX Model Registry — Layer 5.

Maps model names to architecture configs for scratch training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True, slots=True)
class ModelArchConfig:
    name: str
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    vocab_size: int
    max_seq_length: int
    param_count_approx: int


# Built-in registry of common architectures
_REGISTRY: Dict[str, ModelArchConfig] = {
    "llama2-7b": ModelArchConfig(
        "llama2-7b", 4096, 32, 32, 11008, 32000, 4096, 6_738_000_000
    ),
    "llama2-13b": ModelArchConfig(
        "llama2-13b", 5120, 40, 40, 13824, 32000, 4096, 13_015_000_000
    ),
    "mistral-7b": ModelArchConfig(
        "mistral-7b", 4096, 32, 32, 14336, 32000, 8192, 7_241_000_000
    ),
    "phi-2": ModelArchConfig(
        "phi-2", 2560, 32, 32, 10240, 51200, 2048, 2_780_000_000
    ),
    "gpt2-small": ModelArchConfig(
        "gpt2-small", 768, 12, 12, 3072, 50257, 1024, 124_000_000
    ),
    "gpt2-medium": ModelArchConfig(
        "gpt2-medium", 1024, 24, 16, 4096, 50257, 1024, 355_000_000
    ),
    "gpt2-large": ModelArchConfig(
        "gpt2-large", 1280, 36, 20, 5120, 50257, 1024, 774_000_000
    ),
}


class ModelRegistry:
    """Registry for looking up known model architectures."""

    def __init__(self):
        self._custom: Dict[str, ModelArchConfig] = {}

    def get(self, name: str) -> Optional[ModelArchConfig]:
        return self._custom.get(name) or _REGISTRY.get(name)

    def register(self, config: ModelArchConfig) -> None:
        self._custom[config.name] = config

    def list_available(self) -> list:
        return sorted(set(list(_REGISTRY.keys()) + list(self._custom.keys())))
