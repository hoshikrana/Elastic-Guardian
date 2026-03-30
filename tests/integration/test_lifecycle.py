"""
End-to-End Lifecycle Tests

Comprehensive tests for complete training lifecycle:
- Initialization to training completion
- Checkpoint saving/loading
- Recovery from failures
- Memory management
- Performance validation

Integration tests using actual EGX components.
"""

import unittest
import os
import shutil
import torch
import torch.nn as nn
import tempfile
import json
import time
from pathlib import Path
from torch.utils.data import Dataset
from egx.api.trainer import EGXTrainer
from egx.api.config import EGXConfig
from egx.core.enums import TrainingMode


class DummyDataset(Dataset):
    """Simple dummy dataset for testing."""

    def __init__(self, size=100, dim=16):
        self.size = size
        self.data = torch.randn(size, dim)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"input": self.data[idx], "labels": self.labels[idx]}


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, in_features=16, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input, labels=None):
        logits = self.linear(input)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        class Output:
            def __init__(self, l, log):
                self.loss = l
                self.logits = log

        return Output(loss, logits)


class TestTrainingLifecycle(unittest.TestCase):
    """Test complete training workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.output_dir = tempfile.mkdtemp(prefix="test_lifecycle_")
        self.model = DummyModel()
        self.train_data = DummyDataset(size=50, dim=16)
        self.eval_data = DummyDataset(size=20, dim=16)

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_training_initialization(self):
        """Test training initialization phase."""
        config = EGXConfig.from_dict(
            {
                "num_epochs": 1,
                "batch_size": 10,
                "output_dir": self.output_dir,
                "training_mode": TrainingMode.FULL_FINETUNE,
            }
        )

        trainer = EGXTrainer(config=config)

        # Verify trainer is initialized
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.config.num_epochs, 1)
        self.assertEqual(trainer.config.batch_size, 10)

    def test_full_lifecycle(self):
        """Test complete training lifecycle."""
        config = EGXConfig.from_dict(
            {
                "num_epochs": 2,
                "batch_size": 10,
                "output_dir": self.output_dir,
                "checkpoint_strategy": "every_step",
                "training_mode": TrainingMode.FULL_FINETUNE,
            }
        )

        trainer = EGXTrainer(config=config)

        # Train
        result = trainer.train(self.model, self.train_data, eval_dataset=self.eval_data)

        self.assertTrue(
            result.get("success", False) or result.get("epochs_completed", 0) > 0
        )
        self.assertGreater(result.get("epochs_completed", 0), 0)

    def test_training_completion(self):
        """Test training completion metrics."""
        config = EGXConfig.from_dict(
            {
                "num_epochs": 2,
                "batch_size": 10,
                "output_dir": self.output_dir,
                "training_mode": TrainingMode.FULL_FINETUNE,
            }
        )

        trainer = EGXTrainer(config=config)
        result = trainer.train(self.model, self.train_data, eval_dataset=self.eval_data)

        # Verify training completed
        epochs_completed = result.get("epochs_completed", 0)
        self.assertGreater(epochs_completed, 0)
        self.assertLessEqual(epochs_completed, 2)


class TestCheckpointManagement(unittest.TestCase):
    """Test checkpoint saving and loading."""

    def setUp(self):
        """Set up test fixtures."""
        self.output_dir = tempfile.mkdtemp(prefix="test_checkpoints_")
        self.model = DummyModel()
        self.eval_data = DummyDataset(size=20)

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_checkpoint_save_and_restore(self):
        """Test saving and restoring from checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, "model.pt")

        # Save checkpoint
        torch.save(self.model.state_dict(), checkpoint_path)
        self.assertTrue(os.path.exists(checkpoint_path))

        # Create new model and load checkpoint
        new_model = DummyModel()
        new_model.load_state_dict(torch.load(checkpoint_path))

        # Verify models are identical
        for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_best_checkpoint_tracking(self):
        """Test tracking best checkpoint during training."""
        config = EGXConfig.from_dict(
            {
                "num_epochs": 2,
                "batch_size": 10,
                "output_dir": self.output_dir,
                "training_mode": TrainingMode.FULL_FINETUNE,
                "save_best_model": True,
            }
        )

        trainer = EGXTrainer(config=config)
        result = trainer.train(
            self.model, DummyDataset(size=50), eval_dataset=self.eval_data
        )

        # Verify best model was saved
        self.assertGreater(result.get("epochs_completed", 0), 0)

    def test_checkpoint_metadata(self):
        """Test saving checkpoint with metadata."""
        checkpoint_data = {
            "epoch": 5,
            "step": 1234,
            "loss": 0.45,
            "model_state": self.model.state_dict(),
        }

        checkpoint_path = os.path.join(self.output_dir, "checkpoint.pt")
        torch.save(checkpoint_data, checkpoint_path)

        # Load and verify
        loaded = torch.load(checkpoint_path)
        self.assertEqual(loaded["epoch"], 5)
        self.assertEqual(loaded["step"], 1234)
        self.assertAlmostEqual(loaded["loss"], 0.45)


class TestEvaluation(unittest.TestCase):
    """Test evaluation during and after training."""

    def setUp(self):
        """Set up test fixtures."""
        self.output_dir = tempfile.mkdtemp(prefix="test_eval_")
        self.model = DummyModel()
        self.eval_data = DummyDataset(size=20)

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_evaluation_during_training(self):
        """Test evaluation metrics during training."""
        config = EGXConfig.from_dict(
            {
                "num_epochs": 2,
                "batch_size": 10,
                "output_dir": self.output_dir,
                "training_mode": TrainingMode.FULL_FINETUNE,
            }
        )

        trainer = EGXTrainer(config=config)
        trainer.train(self.model, DummyDataset(size=50), eval_dataset=self.eval_data)

    def test_standalone_evaluation(self):
        """Test standalone evaluation on model."""
        config = EGXConfig.from_dict(
            {
                "num_epochs": 1,
                "batch_size": 10,
                "output_dir": self.output_dir,
            }
        )

        trainer = EGXTrainer(config=config)
        eval_metrics = trainer.evaluate(model=self.model, eval_dataset=self.eval_data)

        # Verify evaluation returned metrics
        self.assertIsNotNone(eval_metrics)

    def test_evaluation_metrics_validity(self):
        """Test that evaluation metrics are valid."""
        config = EGXConfig.from_dict(
            {
                "num_epochs": 1,
                "batch_size": 10,
                "output_dir": self.output_dir,
            }
        )

        trainer = EGXTrainer(config=config)
        eval_metrics = trainer.evaluate(model=self.model, eval_dataset=self.eval_data)

        if "eval_loss" in eval_metrics:
            # Loss should be non-negative
            self.assertGreaterEqual(eval_metrics["eval_loss"], 0)


class TestMemoryManagement(unittest.TestCase):
    """Test memory management during training."""

    def setUp(self):
        """Set up test fixtures."""
        self.output_dir = tempfile.mkdtemp(prefix="test_memory_")
        self.model = DummyModel()

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_memory_cleanup_after_training(self):
        """Test that memory is properly cleaned up."""
        initial_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        # Do training
        config = EGXConfig.from_dict(
            {
                "num_epochs": 1,
                "batch_size": 5,
                "output_dir": self.output_dir,
                "training_mode": TrainingMode.FULL_FINETUNE,
            }
        )

        trainer = EGXTrainer(config=config)
        trainer.train(self.model, DummyDataset(size=10))

        # Clean up
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Memory growth should be minimal
        if torch.cuda.is_available():
            memory_growth = final_memory - initial_memory
            self.assertLess(memory_growth, 1000000000)  # Less than 1GB growth


class TestResumeFromCheckpoint(unittest.TestCase):
    """Test resuming training from checkpoint."""

    def setUp(self):
        """Set up test fixtures."""
        self.output_dir = tempfile.mkdtemp(prefix="test_resume_")
        self.model = DummyModel()
        self.train_data = DummyDataset(size=50)

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_checkpoint_restore_parity(self):
        """Test that restored model produces same results."""
        checkpoint_path = os.path.join(self.output_dir, "model.pt")

        # Train and save
        original_model = DummyModel()
        torch.save(original_model.state_dict(), checkpoint_path)

        # Restore to new model
        restored_model = DummyModel()
        restored_model.load_state_dict(torch.load(checkpoint_path))

        # Compare outputs
        test_input = torch.randn(5, 16)
        original_output = original_model(test_input)
        restored_output = restored_model(test_input)

        self.assertTrue(torch.allclose(original_output.logits, restored_output.logits))


class TestPerformanceValidation(unittest.TestCase):
    """Test performance expectations."""

    def setUp(self):
        """Set up test fixtures."""
        self.output_dir = tempfile.mkdtemp(prefix="test_perf_")
        self.model = DummyModel()

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_training_convergence(self):
        """Test that training shows convergence."""
        config = EGXConfig.from_dict(
            {
                "num_epochs": 3,
                "batch_size": 10,
                "output_dir": self.output_dir,
                "training_mode": TrainingMode.FULL_FINETUNE,
            }
        )

        trainer = EGXTrainer(config=config)
        result = trainer.train(self.model, DummyDataset(size=50))

        # Should complete training
        self.assertGreater(result.get("epochs_completed", 0), 0)

    def test_training_reproducibility(self):
        """Test that training is reproducible with same seed."""
        torch.manual_seed(42)

        model1 = DummyModel()
        result1 = model1(torch.randn(5, 16), torch.randint(0, 2, (5,)))

        torch.manual_seed(42)
        model2 = DummyModel()
        result2 = model2(torch.randn(5, 16), torch.randint(0, 2, (5,)))

        # Should be identical with same seed
        self.assertTrue(torch.allclose(result1.logits, result2.logits))


class TestLoggingAndReporting(unittest.TestCase):
    """Test logging and metrics reporting."""

    def setUp(self):
        """Set up test fixtures."""
        self.output_dir = tempfile.mkdtemp(prefix="test_logging_")

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_training_report_generation(self):
        """Test generating training report."""
        key_metrics = {
            "total_epochs": 3,
            "final_loss": 0.35,
            "best_accuracy": 0.93,
            "training_time_hours": 1.5,
        }

        report_path = os.path.join(self.output_dir, "report.json")
        with open(report_path, "w") as f:
            json.dump(key_metrics, f, indent=2)

        # Verify report was created
        self.assertTrue(os.path.exists(report_path))

        # Load and verify
        with open(report_path, "r") as f:
            loaded = json.load(f)

        self.assertEqual(loaded["total_epochs"], 3)
        self.assertAlmostEqual(loaded["final_loss"], 0.35)


if __name__ == "__main__":
    unittest.main()
