"""
PyTorch Dataset classes for polarization classification (binary and multi-label).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset


class PolarizationDataset(Dataset):
    """
    Unified PyTorch Dataset for Polarization detection and categorization tasks.

    Parameters
    ----------
    texts : list of str
        Input text sequences.
    labels : list or np.ndarray, optional
        Target labels (single scalar int/long for binary, list of floats for multi-label).
    tokenizer : PreTrainedTokenizer
        HuggingFace tokenizer instance.
    max_length : int, default=128
        Maximum sequence length for truncation.
    is_multilabel : bool, default=False
        Whether the task is multi-label (torch.float labels) or single-label (torch.long labels).
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[Union[List[Any], Any]] = None,
        tokenizer: Any = None,
        max_length: int = 128,
        is_multilabel: bool = False,
    ):
        self.texts = [str(t) for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_multilabel = is_multilabel

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors="pt",
            )
            item = {key: val.squeeze(0) for key, val in encoding.items()}
        else:
            item = {"text": text}

        if self.labels is not None:
            label = self.labels[idx]
            if self.is_multilabel:
                item["labels"] = torch.tensor(label, dtype=torch.float)
            else:
                item["labels"] = torch.tensor(label, dtype=torch.long)

        return item
