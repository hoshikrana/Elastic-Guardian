"""GPU Validation: Full training loop test."""
import unittest
import torch


class TestFullTrainLoop(unittest.TestCase):

    @staticmethod
    def _sum_loss(outputs, batch):
        """Simple loss function for raw tensor model outputs."""
        return outputs.sum()

    def test_kernel_train_step(self):
        from egx.training.kernel import TrainingKernel
        import torch.nn as nn
        model = nn.Linear(16, 4)
        kernel = TrainingKernel(model=model, optimizer_type="sgd", learning_rate=0.01, loss_fn=self._sum_loss)
        # Use a simple forward that returns a scalar sum for .backward()
        original_forward = model.forward
        model.forward = lambda **kwargs: original_forward(kwargs.get("input", torch.randn(2, 16)))
        batch = {"input": torch.randn(2, 16)}
        loss = kernel.train_step(batch, step=0)
        self.assertIsInstance(loss, float)

    def test_kernel_multiple_steps(self):
        from egx.training.kernel import TrainingKernel
        import torch.nn as nn
        model = nn.Linear(16, 4)
        model.forward = lambda **kwargs: model.__class__.forward(model, kwargs.get("input", torch.randn(2, 16)))
        kernel = TrainingKernel(model=model, optimizer_type="adamw", learning_rate=0.01, loss_fn=self._sum_loss)
        losses = []
        for step in range(5):
            loss = kernel.train_step({"input": torch.randn(2, 16)}, step=step)
            losses.append(loss)
        self.assertEqual(len(losses), 5)

    def test_kernel_with_callbacks(self):
        from egx.training.kernel import TrainingKernel
        import torch.nn as nn
        log = []
        def my_callback(step, loss):
            log.append((step, loss))
        model = nn.Linear(16, 4)
        model.forward = lambda **kwargs: nn.Linear.forward(model, kwargs.get("input", torch.randn(2, 16)))
        kernel = TrainingKernel(model=model, optimizer_type="sgd", callbacks=[my_callback], loss_fn=self._sum_loss)
        kernel.train_step({"input": torch.randn(2, 16)}, step=0)
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0][0], 0)


if __name__ == "__main__":
    unittest.main()
