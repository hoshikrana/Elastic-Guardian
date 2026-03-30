"""
EGX Layer 1 Hardening Tests.

Verifies:
1. Law 10 (int bytes, bool-trap)
2. Law 5 (immutable frozen contracts)
3. Exception recoverability
"""

import unittest
from egx.core.exceptions import BoolAsIntError
from egx.core.memory.validators import StandardMemoryValidator
from egx.core.memory.value import MemoryValue
from egx.core.models import GPUSpec
from egx.core.enums import HardwareTier


class TestLayer1Hardening(unittest.TestCase):

    def test_law_10_bool_trap(self):
        """Verifies that passing True as memory value raises BoolAsIntError."""
        with self.assertRaises(BoolAsIntError):
            StandardMemoryValidator.validate(True, "test_field")

    def test_memory_value_immutable(self):
        """Verifies MemoryValue slots and immutability."""
        mv = MemoryValue(1024)
        with self.assertRaises(AttributeError):
            mv.bytes = 2048  # type: ignore

    def test_contract_frozen(self):
        """Verifies that public contracts are frozen."""
        spec = GPUSpec(
            device_id=0,
            name="Test GPU",
            vram_bytes=1000,
            compute_capability=(8, 0),
            memory_bandwidth_gbps=100.0,
            fp16_tflops=10.0,
            bf16_tflops=10.0,
            supports_flash_attn2=True,
            supports_fp8=False,
            nvlink_peer_ids=(),
        )
        with self.assertRaises(AttributeError):
            spec.vram_bytes = 2000  # type: ignore

    def test_hardware_tier_logic(self):
        """Verifies tier calculation from int bytes."""
        spec = GPUSpec(
            device_id=0,
            name="Titan",
            vram_bytes=24 * 1024 * 1024 * 1024,
            compute_capability=(8, 6),
            memory_bandwidth_gbps=900.0,
            fp16_tflops=30.0,
            bf16_tflops=30.0,
            supports_flash_attn2=True,
            supports_fp8=False,
            nvlink_peer_ids=(),
        )
        self.assertEqual(spec.tier, HardwareTier.WORKSTATION)


if __name__ == "__main__":
    unittest.main()
