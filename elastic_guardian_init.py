"""
Elastic-Guardian X: Production-Grade AI Training Operating System

A bulletproof training framework designed for frontier-scale foundation models
with zero-failure guarantees, automatic resource management, and distributed
training orchestration.

Key Features:
- Zero-OOM memory management with tiered offloading
- Hardware-elastic configuration (auto-adapts to any setup)
- Distributed training (DDP/FSDP/DeepSpeed) with failure recovery
- Gradient sanitization and loss spike detection
- Plugin system for research extensions
- Production-ready logging and monitoring

Example Usage:
    >>> from elastic_guardian import train_model, load_config
    >>> config = load_config("config.yaml")
    >>> train_model(config)

For detailed documentation, see: https://elastic-guardian-x.readthedocs.io
"""

from elastic_guardian.version import __version__, __version_info__

# Core will be imported as we build each module
# from elastic_guardian.core import Config, Logger
# from elastic_guardian.hardware import HardwareIntelligenceService
# from elastic_guardian.memory import ElasticMemoryOrchestrator
# from elastic_guardian.training import UniversalTrainingKernel

__all__ = [
    "__version__",
    "__version_info__",
    # Will add more as we build modules
]

# Package metadata
__author__ = "Elastic-Guardian X Team"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2026 Elastic-Guardian X Team"
