"""
Evaluation metrics for binary and multi-label polarization classification.
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics_binary(eval_pred: Any) -> Dict[str, float]:
    """
    Compute evaluation metrics for binary polarization classification (Subtask 1).

    Parameters
    ----------
    eval_pred : EvalPrediction or tuple of (predictions, labels)

    Returns
    -------
    dict with accuracy, precision, recall, f1, and f1_macro.
    """
    if hasattr(eval_pred, "predictions"):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
    else:
        predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds = np.argmax(predictions, axis=1)

    acc = float(accuracy_score(labels, preds))
    prec = float(precision_score(labels, preds, average="binary", zero_division=0))
    rec = float(recall_score(labels, preds, average="binary", zero_division=0))
    f1 = float(f1_score(labels, preds, average="binary", zero_division=0))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f1_macro": macro_f1,
    }


def compute_metrics_multi_label(eval_pred: Any, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute evaluation metrics for multi-label classification (Subtasks 2 & 3).

    Parameters
    ----------
    eval_pred : EvalPrediction or tuple of (predictions, labels)
    threshold : float, default=0.5
        Probability threshold for positive binary decision.

    Returns
    -------
    dict with f1_macro and f1_micro.
    """
    if hasattr(eval_pred, "predictions"):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
    else:
        logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    # Numerically stable sigmoid
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    predictions = (probs > threshold).astype(int)
    labels = np.asarray(labels, dtype=int)

    f1_macro = float(f1_score(labels, predictions, average="macro", zero_division=0))
    f1_micro = float(f1_score(labels, predictions, average="micro", zero_division=0))

    return {
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
    }
