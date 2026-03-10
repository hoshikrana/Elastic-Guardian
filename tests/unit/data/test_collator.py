"""Unit tests for DynamicPaddingCollator."""
import unittest
import torch


class TestDynamicPaddingCollator(unittest.TestCase):
    def test_pads_to_max_length(self):
        from egx.data.collator import DynamicPaddingCollator
        collator = DynamicPaddingCollator(pad_token_id=0)
        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5, 6, 7, 8], "attention_mask": [1, 1, 1, 1, 1]},
        ]
        batch = collator(features)
        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertTrue(batch["input_ids"].shape[1] >= 5)

    def test_pad_to_multiple_of(self):
        from egx.data.collator import DynamicPaddingCollator
        collator = DynamicPaddingCollator(pad_to_multiple_of=8)
        features = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}]
        batch = collator(features)
        self.assertEqual(batch["input_ids"].shape[1] % 8, 0)

    def test_labels_padded_with_ignore(self):
        from egx.data.collator import DynamicPaddingCollator
        collator = DynamicPaddingCollator(pad_token_id=0, pad_to_multiple_of=8)
        features = [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [10, 20]},
            {"input_ids": [3, 4, 5], "attention_mask": [1, 1, 1], "labels": [30, 40, 50]},
        ]
        batch = collator(features)
        # Labels should have -100 for padding
        self.assertTrue((batch["labels"][0] == -100).any())

    def test_max_seq_len_truncation(self):
        from egx.data.collator import DynamicPaddingCollator
        collator = DynamicPaddingCollator(max_seq_len=4, pad_to_multiple_of=1)
        features = [{"input_ids": list(range(10)), "attention_mask": [1]*10}]
        batch = collator(features)
        self.assertLessEqual(batch["input_ids"].shape[1], 4)


if __name__ == "__main__":
    unittest.main()
