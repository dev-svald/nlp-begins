# Multilingual Online Polarization Detection (NLP-Begins)

[![CI](https://github.com/dev-svald/nlp-begins/actions/workflows/ci.yml/badge.svg)](https://github.com/dev-svald/nlp-begins/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A deep learning framework for **multilingual, multicultural, and multi-event online polarization detection** in **English** and low-resource African languages (**Swahili**).

The framework implements and benchmarks state-of-the-art transformer architectures—including **AfriBERTa**, **SwahBERT**, **Afro-XLM-R**, and **XLM-RoBERTa**—across three distinct classification subtasks with specialized loss weighting for severe multi-label class imbalance.

---

## Table of Contents

- [Overview & Subtasks](#overview--subtasks)
  - [Subtask 1: Binary Polarization Detection](#subtask-1-binary-polarization-detection)
  - [Subtask 2: Polarization Type Classification](#subtask-2-polarization-type-classification)
  - [Subtask 3: Polarization Manifestation Identification](#subtask-3-polarization-manifestation-identification)
- [Pretrained Models Supported](#pretrained-models-supported)
- [Methodology & Architecture](#methodology--architecture)
  - [Text Cleaning Strategies](#text-cleaning-strategies)
  - [Class-Weighted Multi-Label Loss](#class-weighted-multi-label-loss)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset Organization](#dataset-organization)
- [Usage](#usage)
  - [1. Running via CLI](#1-running-via-cli)
  - [2. Python API Usage](#2-python-api-usage)
  - [3. Interactive Notebooks](#3-interactive-notebooks)
- [Testing](#testing)
- [Benchmark Results](#benchmark-results)
- [License & Acknowledgements](#license--acknowledgements)

---

## Overview & Subtasks

Online polarization manifests differently across cultural, linguistic, and regional contexts. This project systematically addresses three core challenges:

### Subtask 1: Binary Polarization Detection
Determines whether an online post contains polarized content:
- **Target**: `polarization` $\in \{0, 1\}$ (0: Not Polarized, 1: Polarized)
- **Evaluation Metric**: Macro F1 Score
- **Experiments**: Comparing raw text versus emoji-preservation (demojize) versus emoji-deletion strategies.

### Subtask 2: Polarization Type Classification
Multi-label classification identifying the target community or topic:
- **Labels**: `political`, `racial/ethnic`, `religious`, `gender/sexual`, `other`
- **Challenge**: Extreme class imbalance across target categories.
- **Solution**: Positive class-weighted Binary Cross-Entropy loss ($pos\_weight$).

### Subtask 3: Polarization Manifestation Identification
Multi-label classification determining *how* the polarization is expressed:
- **Labels**: `vilification`, `extreme_language`, `stereotype`, `invalidation`, `lack_of_empathy`, `dehumanization`
- **Output**: 6-dimensional binary prediction vector.

---

## Pretrained Models Supported

| Model Alias | HuggingFace Checkpoint | Language Scope | Description |
| :--- | :--- | :--- | :--- |
| `afriberta` | `castorini/afriberta_large` | African Multilingual | State-of-the-art transformer for African languages (including Swahili) |
| `swahbert` | `flax-community/swahbert-base` | Swahili Monolingual | BERT pretrained natively on large-scale Swahili corpora |
| `afro-xlm-r`| `Davlan/afro-xlm-roberta-base` | African Multilingual | Cross-lingual RoBERTa fine-tuned on African languages |
| `xlm-roberta`| `xlm-roberta-base` | Massive Multilingual | 100-language cross-lingual RoBERTa |
| `roberta` | `roberta-base` | English Monolingual | Standard English RoBERTa baseline |
| `distilroberta`| `distilroberta-base` | English Monolingual | Distilled, lightweight RoBERTa |
| `bert` | `bert-base-uncased` | English Monolingual | Standard BERT uncased baseline |

---

## Methodology & Architecture

### Text Cleaning Strategies

1. **Strategy 0 (Raw)**: Preserves text as-is.
2. **Strategy 1 (Emoji-to-Text)**: Strips URLs and `@user` handles; converts emojis into descriptive English/Swahili text tokens via `emoji.demojize` to preserve emotional nuance.
3. **Strategy 2 (Emoji Deletion)**: Strips URLs, `@user` handles, and deletes all Unicode emoji characters for concise representation.

### Class-Weighted Multi-Label Loss

In multi-label classification, positive samples for rare categories (e.g. `gender/sexual` or `dehumanization`) are drastically outnumbered by negative samples. To combat this gradient vanishing issue, positive class weights $w_c$ are computed via inverse class frequency:

$$w_c = \frac{N_{\text{total}}}{N_c^+ + \epsilon}$$

$$\tilde{w}_c = \frac{w_c}{\frac{1}{C}\sum_{j=1}^C w_j}$$

The loss function is implemented as a custom PyTorch module (`CustomWeightedLossTrainer`) wrapping:

$$\mathcal{L}_{\text{weighted}}(x, y) = - \sum_{c=1}^C \left[ \tilde{w}_c \cdot y_c \log \sigma(x_c) + (1 - y_c) \log (1 - \sigma(x_c)) \right]$$

---

## Repository Structure

```text
nlp-begins/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions continuous integration
├── data/                        # Local dataset storage (ignored by git)
│   └── .gitkeep
├── notebooks/                   # Interactive Jupyter & Colab practicals
│   └── 01_multilingual_polarization_detection.ipynb
├── scripts/                     # Standalone CLI execution entry points
│   ├── run_subtask1.py          # Train & evaluate Subtask 1 (Binary)
│   ├── run_subtask2.py          # Train & evaluate Subtask 2 (Multi-label types)
│   ├── run_subtask3.py          # Train & evaluate Subtask 3 (Manifestations)
│   └── legacy_colab_pipeline.py # Preserved original pipeline script
├── src/
│   └── polarization/            # Modular Python package
│       ├── __init__.py
│       ├── dataset.py           # PyTorch Dataset for single & multi-label tasks
│       ├── loss.py              # CustomWeightedLossTrainer & pos_weight calculator
│       ├── metrics.py           # Binary & multi-label Macro/Micro F1, precision, recall
│       ├── models.py            # Pretrained transformer model factory & registry
│       ├── preprocessing.py     # Multi-strategy text cleaning & emoji normalizers
│       ├── training.py          # HuggingFace Trainer pipelines & reproducible seeding
│       └── visualization.py     # EDA class distributions, heatmaps & model bar charts
├── tests/                       # Unit test suite
│   ├── __init__.py
│   ├── test_dataset.py
│   ├── test_metrics.py
│   └── test_preprocessing.py
├── .gitignore
├── LICENSE                      # MIT License
├── pyproject.toml               # Modern build specification & dependencies
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/dev-svald/nlp-begins.git
cd nlp-begins

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

---

## Dataset Organization

Place the competition data partitions under `data/` or corresponding task directories:

```text
data/
├── subtask1/
│   ├── train/ (eng.csv, swa.csv)
│   └── dev/   (eng.csv, swa.csv)
├── subtask2/
│   ├── train/ (eng.csv, swa.csv)
│   └── dev/   (eng.csv, swa.csv)
└── subtask3/
    └── train/ (eng.csv)
```

---

## Usage

### 1. Running via CLI

#### Subtask 1: Binary Polarization Detection (English with RoBERTa)
```bash
python scripts/run_subtask1.py \
    --data-path data/subtask1/train/eng.csv \
    --model roberta \
    --cleaning-strategy 1 \
    --epochs 3 \
    --batch-size 16 \
    --lr 2e-5
```

#### Subtask 1: Swahili with AfriBERTa
```bash
python scripts/run_subtask1.py \
    --data-path data/subtask1/train/swa.csv \
    --model afriberta \
    --cleaning-strategy 1 \
    --epochs 3 \
    --batch-size 16 \
    --lr 2e-5
```

#### Subtask 2: Multi-Label Polarization Types (Weighted Loss)
```bash
python scripts/run_subtask2.py \
    --data-path data/subtask2/train/eng.csv \
    --model xlm-roberta \
    --epochs 4 \
    --lr 1e-5
```

#### Subtask 3: Manifestation Identification
```bash
python scripts/run_subtask3.py \
    --data-path data/subtask3/train/eng.csv \
    --model bert \
    --epochs 3 \
    --batch-size 8 \
    --lr 2e-5
```

### 2. Python API Usage

```python
from polarization.preprocessing import clean_text
from polarization.models import load_model_and_tokenizer
from polarization.dataset import PolarizationDataset
from polarization.training import train_subtask1

# 1. Clean input texts
cleaned = clean_text("Check this out @user https://example.com 🔥", strategy=1)

# 2. Instantiate tokenizer and model
tokenizer, model = load_model_and_tokenizer("afriberta", num_labels=2)

# 3. Create datasets
train_ds = PolarizationDataset(texts=train_texts, labels=train_labels, tokenizer=tokenizer)
val_ds = PolarizationDataset(texts=val_texts, labels=val_labels, tokenizer=tokenizer)

# 4. Train pipeline
trainer, eval_metrics = train_subtask1(model, tokenizer, train_ds, val_ds)
print("Validation Macro F1:", eval_metrics["eval_f1_macro"])
```

### 3. Interactive Notebooks

To run the interactive step-by-step EDA and full experiment suite:

```bash
jupyter lab notebooks/01_multilingual_polarization_detection.ipynb
```

---

## Testing

Run the automated unit test suite:

```bash
# Using Python's standard unittest runner:
python -m unittest discover -s tests

# Or using pytest:
pytest tests/
```

---

## Benchmark Results

Validation Macro F1 scores achieved across model configurations:

| Task | Language | Architecture | Macro F1 |
| :--- | :--- | :--- | :--- |
| **Subtask 1 (Polarization)** | Swahili | **AfriBERTa (Large)** | **0.797** |
| **Subtask 1 (Polarization)** | Swahili | **SwahBERT** | **0.751** |
| **Subtask 1 (Polarization)** | English | **XLM-RoBERTa** | **0.784** |
| **Subtask 1 (Polarization)** | English | **RoBERTa** | **0.776** |
| **Subtask 2 (Types)** | English | **DistilRoBERTa (Weighted Loss)** | **0.732** |
| **Subtask 2 (Types)** | Swahili | **AfriBERTa (Weighted Loss)** | **0.748** |

---

## License & Acknowledgements

- **License**: Released under the [MIT License](LICENSE).
- **Author**: [`dev-svald`](https://github.com/dev-svald).
- **Upstream Source**: Forked and restructured from [`naoussijames/NLP-Begins`](https://github.com/naoussijames/NLP-Begins).
