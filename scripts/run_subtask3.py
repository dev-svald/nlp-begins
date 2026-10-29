"""
CLI script to train and evaluate Subtask 3: Manifestation Identification (Multi-Label).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
from sklearn.model_selection import train_test_split

from polarization.dataset import PolarizationDataset
from polarization.models import load_model_and_tokenizer
from polarization.preprocessing import clean_text
from polarization.training import train_subtask2

DEFAULT_MANIFESTATIONS = [
    "vilification",
    "extreme_language",
    "stereotype",
    "invalidation",
    "lack_of_empathy",
    "dehumanization",
]


def main():
    parser = argparse.ArgumentParser(description="Subtask 3: Polarization Manifestation Identification")
    parser.add_argument("--data-path", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--model", type=str, default="bert", help="Model name or alias")
    parser.add_argument("--manifestations", nargs="+", default=DEFAULT_MANIFESTATIONS, help="Manifestation labels")
    parser.add_argument("--cleaning-strategy", type=int, default=1, choices=[0, 1, 2], help="Cleaning strategy")
    parser.add_argument("--text-col", type=str, default="text", help="Text column name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--output-dir", type=str, default="./results/subtask3", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at '{args.data_path}'")
        sys.exit(1)

    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df):,} samples.")

    if args.cleaning_strategy > 0:
        print(f"Applying text cleaning strategy {args.cleaning_strategy}...")
        df["cleaned_text"] = df[args.text_col].apply(lambda t: clean_text(t, strategy=args.cleaning_strategy))
        text_column = "cleaned_text"
    else:
        text_column = args.text_col

    train_df, val_df = train_test_split(df, test_size=args.val_split, random_state=42)
    print(f"Train samples: {len(train_df):,} | Val samples: {len(val_df):,}")

    print(f"Loading model '{args.model}' with {len(args.manifestations)} manifestation classes...")
    tokenizer, model = load_model_and_tokenizer(
        args.model,
        num_labels=len(args.manifestations),
        is_multilabel=True
    )

    train_dataset = PolarizationDataset(
        texts=train_df[text_column].tolist(),
        labels=train_df[args.manifestations].values.tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
        is_multilabel=True
    )
    val_dataset = PolarizationDataset(
        texts=val_df[text_column].tolist(),
        labels=val_df[args.manifestations].values.tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
        is_multilabel=True
    )

    print("Beginning multi-label manifestation training...")
    trainer, eval_results = train_subtask2(
        model_or_wrapper=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    print("Evaluation Results:")
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
