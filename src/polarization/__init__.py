"""
Multilingual, Multicultural and Multievent Online Polarization Detection (English & Swahili).
"""

from polarization.dataset import PolarizationDataset
from polarization.loss import CustomWeightedLossTrainer, calculate_label_weights
from polarization.metrics import compute_metrics_binary, compute_metrics_multi_label
from polarization.models import MODEL_REGISTRY, load_model_and_tokenizer, resolve_model_id
from polarization.preprocessing import (
    calculate_text_length,
    clean_text,
    clean_text_strategy_1,
    clean_text_strategy_2,
)
from polarization.training import set_seed, train_subtask1, train_subtask2

__version__ = "1.0.0"

__all__ = [
    "PolarizationDataset",
    "calculate_label_weights",
    "CustomWeightedLossTrainer",
    "compute_metrics_binary",
    "compute_metrics_multi_label",
    "MODEL_REGISTRY",
    "resolve_model_id",
    "load_model_and_tokenizer",
    "clean_text",
    "clean_text_strategy_1",
    "clean_text_strategy_2",
    "calculate_text_length",
    "set_seed",
    "train_subtask1",
    "train_subtask2",
]
