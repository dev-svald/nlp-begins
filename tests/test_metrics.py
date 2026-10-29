"""
Unit tests for evaluation metrics and class weighting.
"""

import unittest
import numpy as np
import pandas as pd
import torch

from polarization.loss import calculate_label_weights
from polarization.metrics import compute_metrics_binary, compute_metrics_multi_label


class TestMetrics(unittest.TestCase):
    def test_compute_metrics_binary_perfect(self):
        # 4 samples: perfect prediction
        logits = np.array([
            [2.5, -1.0],  # argmax 0
            [-1.2, 3.1],  # argmax 1
            [3.0, -2.0],  # argmax 0
            [-0.5, 2.0],  # argmax 1
        ])
        labels = np.array([0, 1, 0, 1])

        results = compute_metrics_binary((logits, labels))

        self.assertEqual(results["accuracy"], 1.0)
        self.assertEqual(results["f1_macro"], 1.0)

    def test_compute_metrics_multi_label(self):
        # 3 samples, 3 classes
        logits = np.array([
            [3.0, -4.0, 2.5],   # sigmoid -> [1, 0, 1]
            [-2.0, 4.0, -1.5],  # sigmoid -> [0, 1, 0]
            [-3.0, -3.0, -3.0], # sigmoid -> [0, 0, 0]
        ])
        labels = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [0, 0, 0],
        ])

        results = compute_metrics_multi_label((logits, labels), threshold=0.5)

        self.assertEqual(results["f1_macro"], 1.0)
        self.assertEqual(results["f1_micro"], 1.0)

    def test_calculate_label_weights(self):
        # 10 samples: class 0 is rare (1 positive), class 1 is common (9 positives)
        df = pd.DataFrame({
            "col_a": [1] + [0] * 9,
            "col_b": [1] * 9 + [0],
        })

        weights = calculate_label_weights(df, ["col_a", "col_b"])

        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(weights.shape[0], 2)
        # Rare class should receive strictly higher weight
        self.assertGreater(weights[0].item(), weights[1].item())


if __name__ == "__main__":
    unittest.main()
