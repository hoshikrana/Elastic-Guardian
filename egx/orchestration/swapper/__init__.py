"""
EGX Orchestration Swappers — Layer 4.

Provides specialized swappers for VRAM, RAM, and NVMe movement.
"""
from egx.orchestration.swapper.vram_to_ram import VRAMToRAMSwapper
from egx.orchestration.swapper.ram_to_nvme import RAMToNVMeSwapper

__all__ = ["VRAMToRAMSwapper", "RAMToNVMeSwapper"]
