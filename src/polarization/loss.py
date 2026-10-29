"""
Custom loss functions and class weighting strategies for imbalanced multi-label classification.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss


def calculate_label_weights(
    df: pd.DataFrame,
    label_cols: List[str],
    device: Optional[torch.device] = None,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Calculate inverse class frequency weights (pos_weight) for multi-label BCE loss.

    Formula:
        pos_weight = total_samples / (positive_counts + epsilon)
        normalized = pos_weight / mean(pos_weight)

    Parameters
    ----------
    df : pd.DataFrame
        Training dataframe containing label columns with 0 or 1 values.
    label_cols : list of str
        Names of the binary target columns.
    device : torch.device, optional
        Target device for the returned tensor.
    epsilon : float, default=1e-6
        Smoothing factor to prevent division by zero.

    Returns
    -------
    torch.Tensor of shape (num_labels,)
    """
    positive_counts = df[label_cols].sum(axis=0).values.astype(np.float64)
    total_samples = len(df)

    weights = total_samples / (positive_counts + epsilon)
    weights = weights / np.mean(weights)

    tensor_weights = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        tensor_weights = tensor_weights.to(device)
    return tensor_weights


class CustomWeightedLossTrainer(nn.Module):
    """
    PyTorch wrapper module implementing BCEWithLogitsLoss with positive class weights
    to combat severe label imbalance in multi-label classification.

    Parameters
    ----------
    model : nn.Module
        Underlying HuggingFace SequenceClassification model.
    pos_weight : torch.Tensor
        Positive class weights for BCEWithLogitsLoss.
    """

    def __init__(self, model: nn.Module, pos_weight: torch.Tensor):
        super().__init__()
        self.model = model
        self.loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = outputs.logits

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return (loss, logits) if loss is not None else outputs

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
