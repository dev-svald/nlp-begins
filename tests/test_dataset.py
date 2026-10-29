"""
Unit tests for PolarizationDataset class.
"""

import unittest
import torch
from polarization.dataset import PolarizationDataset


class DummyTokenizer:
    """Mock tokenizer returning PyTorch tensor dict for unit testing without network calls."""
    def __call__(self, text, truncation=True, padding=False, max_length=128, return_tensors="pt"):
        words = text.split()[:max_length - 2]
        tokens = [101] + [abs(hash(w)) % 1000 + 1 for w in words] + [102]
        return {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.tensor([[1] * len(tokens)], dtype=torch.long),
        }


class TestPolarizationDataset(unittest.TestCase):
    def setUp(self):
        self.texts = ["First sample sentence.", "Second polarized post online.", "Neutral third post."]
        self.tokenizer = DummyTokenizer()

    def test_single_label_dataset(self):
        labels = [0, 1, 0]
        ds = PolarizationDataset(
            texts=self.texts,
            labels=labels,
            tokenizer=self.tokenizer,
            is_multilabel=False
        )

        self.assertEqual(len(ds), 3)
        item0 = ds[0]

        self.assertIn("input_ids", item0)
        self.assertIn("attention_mask", item0)
        self.assertIn("labels", item0)
        self.assertEqual(item0["labels"].dtype, torch.long)
        self.assertEqual(item0["labels"].item(), 0)

    def test_multi_label_dataset(self):
        labels = [
            [1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]
        ds = PolarizationDataset(
            texts=self.texts,
            labels=labels,
            tokenizer=self.tokenizer,
            is_multilabel=True
        )

        self.assertEqual(len(ds), 3)
        item1 = ds[1]

        self.assertEqual(item1["labels"].dtype, torch.float)
        self.assertEqual(item1["labels"].shape[0], 5)
        self.assertEqual(item1["labels"][1].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
