"""
EGX Test: core/enums.py
"""

import unittest
from egx.core.enums import DeviceType, TrainingMode, DType, HardwareTier, RecoveryAction


class TestCoreEnums(unittest.TestCase):
    def test_device_type_values(self):
        self.assertEqual(DeviceType.CUDA.value, "cuda")
        self.assertEqual(DeviceType.CPU.value, "cpu")

    def test_training_mode_values(self):
        modes = [TrainingMode.FULL_FINETUNE, TrainingMode.LORA, TrainingMode.QLORA]
        self.assertEqual(len(set(modes)), 3)

    def test_dtype_byte_size(self):
        self.assertEqual(DType.FP32.byte_size(), 4)
        self.assertEqual(DType.FP16.byte_size(), 2)
        self.assertEqual(DType.BF16.byte_size(), 2)
        self.assertEqual(DType.INT8.byte_size(), 1)

    def test_hardware_tier(self):
        self.assertIn(HardwareTier.DATACENTER, HardwareTier)

    def test_recovery_action(self):
        self.assertIn(RecoveryAction.HALVE_BATCH, RecoveryAction)
        self.assertIn(RecoveryAction.RETRY, RecoveryAction)


if __name__ == "__main__":
    unittest.main()
