"""Benchmark: State dict serialization speed."""
import time
import tempfile
import unittest
import torch
import torch.nn as nn


class TestExportSpeed(unittest.TestCase):
    def test_state_dict_save_speed(self):
        model = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 512), nn.Linear(512, 256))
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            start = time.perf_counter()
            torch.save(model.state_dict(), f.name)
            elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 5.0, f"Save took {elapsed:.2f}s")

    def test_lora_merger_speed(self):
        from egx.export.lora_merger import LoRAExportMerger
        merger = LoRAExportMerger()
        model = nn.Linear(64, 64)
        start = time.perf_counter()
        merger.merge_and_export(model, "/tmp/egx_bench_export")
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 5.0, f"LoRA merge+export took {elapsed:.2f}s")


if __name__ == "__main__":
    unittest.main()
