"""Unit tests for ThroughputTracker (or training monitoring)."""

import unittest


class TestThroughputTracker(unittest.TestCase):
    def test_import_training_kernel(self):
        from egx.training.kernel import TrainingKernel

        self.assertIsNotNone(TrainingKernel)

    def test_kernel_inherits_base(self):
        from egx.training.kernel import TrainingKernel
        from egx.core.interfaces import BaseTrainingKernel

        self.assertTrue(issubclass(TrainingKernel, BaseTrainingKernel))

    def test_kernel_accepts_callbacks(self):
        import torch.nn as nn
        from egx.training.kernel import TrainingKernel

        model = nn.Linear(16, 4)
        calls = []
        kernel = TrainingKernel(model=model, callbacks=[lambda s, l: calls.append(s)])
        self.assertEqual(len(kernel.callbacks), 1)

    def test_kernel_scheduler_options(self):
        import torch.nn as nn
        from egx.training.kernel import TrainingKernel

        model = nn.Linear(16, 4)
        k_linear = TrainingKernel(model=model, scheduler_type="linear", warmup_steps=10)
        self.assertIsNotNone(k_linear.scheduler)
        k_cosine = TrainingKernel(model=model, scheduler_type="cosine")
        self.assertIsNotNone(k_cosine.scheduler)
        k_none = TrainingKernel(model=model, scheduler_type=None)
        self.assertIsNone(k_none.scheduler)


if __name__ == "__main__":
    unittest.main()
