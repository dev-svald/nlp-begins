"""
Training pipelines for Subtasks 1, 2, and 3 using HuggingFace Trainer.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from polarization.metrics import compute_metrics_binary, compute_metrics_multi_label


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_subtask1(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    val_dataset: Any,
    output_dir: str = "./results/subtask1",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    seed: int = 42,
) -> Tuple[Trainer, Dict[str, float]]:
    """
    Train and evaluate binary polarization detection model (Subtask 1).
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        seed=seed,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_binary,
        data_collator=data_collator,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    return trainer, eval_results


def train_subtask2(
    model_or_wrapper: Any,
    tokenizer: Any,
    train_dataset: Any,
    val_dataset: Any,
    output_dir: str = "./results/subtask2",
    epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    seed: int = 42,
) -> Tuple[Trainer, Dict[str, float]]:
    """
    Train and evaluate multi-label polarization type classification (Subtask 2)
    using class-imbalance weighted loss.
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        seed=seed,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model_or_wrapper,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_multi_label,
        data_collator=data_collator,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    return trainer, eval_results
