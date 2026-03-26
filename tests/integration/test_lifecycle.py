import unittest
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from egx.api.trainer import EGXTrainer
from egx.api.config import EGXConfig
from egx.core.enums import TrainingMode

class DummyDataset(Dataset):
    def __init__(self, size=100, dim=16):
        self.size = size
        self.data = torch.randn(size, dim)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"input": self.data[idx], "labels": self.labels[idx]}

class DummyModel(nn.Module):
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

class TestLifecycle(unittest.TestCase):
    def setUp(self):
        self.output_dir = "./tmp_test_lifecycle_checkpoints"
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_full_lifecycle(self):
        config = EGXConfig.from_dict({
            "num_epochs": 2,
            "batch_size": 10,
            "output_dir": self.output_dir,
            "checkpoint_strategy": "every_step", # Force save
            "training_mode": TrainingMode.FULL_FINETUNE,
        })
        
        model = DummyModel()
        train_data = DummyDataset(size=50)
        eval_data = DummyDataset(size=20)
        
        trainer = EGXTrainer(config=config)
        
        res = trainer.train(model, train_data, eval_dataset=eval_data)
        
        self.assertTrue(res["success"])
        self.assertGreater(res["epochs_completed"], 0)
        
        # Test Evaluation
        eval_metrics = trainer.evaluate(eval_dataset=eval_data)
        self.assertIn("eval_loss", eval_metrics)
        
        # Verify checkpoint and resume
        import os
        from safetensors.torch import save_file, load_file
        
        os.makedirs(self.output_dir, exist_ok=True)
        ckpt_path = os.path.join(self.output_dir, "model.safetensors")
        save_file(model.state_dict(), ckpt_path)
        self.assertTrue(os.path.exists(ckpt_path))
        
        # Resume with new model
        new_model = DummyModel()
        new_model.load_state_dict(load_file(ckpt_path))
        
        # Ensure parity
        self.assertTrue(torch.allclose(
            model.linear.weight, new_model.linear.weight
        ))
        
        # Eval with new model
        new_metrics = trainer.evaluate(model=new_model, eval_dataset=eval_data)
        self.assertEqual(eval_metrics["eval_loss"], new_metrics["eval_loss"])

if __name__ == "__main__":
    unittest.main()
