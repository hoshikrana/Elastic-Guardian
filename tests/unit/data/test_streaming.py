"""Unit tests for ElasticDataset streaming."""

import unittest
import tempfile
import json
import os


class TestElasticDataset(unittest.TestCase):
    def test_jsonl_streaming(self):
        from egx.data.streaming import ElasticDataset

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for i in range(5):
                json.dump({"text": f"Hello world sentence {i}"}, f)
                f.write("\n")
            path = f.name

        class FakeTokenizer:
            def __call__(self, text, **kwargs):
                import torch

                return {"input_ids": torch.tensor([[1, 2, 3]])}

        ds = ElasticDataset(data_path=path, tokenizer=FakeTokenizer(), max_seq_len=128)
        items = list(ds)
        self.assertEqual(len(items), 5)
        os.unlink(path)

    def test_unsupported_format_raises(self):
        from egx.data.streaming import ElasticDataset

        ds = ElasticDataset(data_path="nonexistent.csv", tokenizer=None)
        with self.assertRaises(ValueError):
            list(ds)

    def test_empty_text_skipped(self):
        from egx.data.streaming import ElasticDataset

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"text": ""}, f)
            f.write("\n")
            json.dump({"text": "hello"}, f)
            f.write("\n")
            path = f.name

        class FakeTokenizer:
            def __call__(self, text, **kwargs):
                import torch

                return {"input_ids": torch.tensor([[1, 2, 3]])}

        ds = ElasticDataset(data_path=path, tokenizer=FakeTokenizer())
        items = list(ds)
        self.assertEqual(len(items), 1)
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()
