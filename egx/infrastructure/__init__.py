# infrastructure — Layer 2. Hardware access. torch + pynvml allowed only here and L5+.
"""
egx.infrastructure — Layer 2. Hardware access.

torch and pynvml are allowed ONLY in this layer and Layer 5+.
All other layers must never import torch or pynvml.

Public surface:
    probe_gpus()       — discover GPUs, returns list[GPUSpec]
    build_topology()   — assemble HardwareTopology from GPUSpecs
"""

from egx.infrastructure.gpu_probe import probe_gpus
from egx.infrastructure.topology_builder import build_topology

__all__ = ["probe_gpus", "build_topology"]
