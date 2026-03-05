"""
EGX Base Exporter — Layer 5.

Abstract interface for all model exporters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger("egx.export")


class BaseExporter(ABC):
    """Interface for model export strategies."""

    @abstractmethod
    def export(self, model: Any, output_path: Path, **kwargs) -> Path:
        """Export a model to the given path. Returns the final output path."""
        ...

    @abstractmethod
    def validate(self, output_path: Path) -> bool:
        """Validate that the exported artifact is loadable."""
        ...

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Human-readable name of the export format."""
        ...

    def _ensure_dir(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
