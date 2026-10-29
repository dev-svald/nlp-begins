"""
Model architectures, pretrained checkpoints, and initialization factory.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "xlm-roberta": {
        "hf_id": "xlm-roberta-base",
        "language": "multilingual",
        "description": "Multilingual RoBERTa supporting both English and Swahili",
    },
    "roberta": {
        "hf_id": "roberta-base",
        "language": "english",
        "description": "Standard English RoBERTa baseline",
    },
    "distilroberta": {
        "hf_id": "distilroberta-base",
        "language": "english",
        "description": "Lightweight English DistilRoBERTa",
    },
    "afro-xlm-r": {
        "hf_id": "Davlan/afro-xlm-roberta-base",
        "language": "african_multilingual",
        "description": "Afro-centric multilingual RoBERTa covering Swahili",
    },
    "swahbert": {
        "hf_id": "flax-community/swahbert-base",
        "language": "swahili",
        "description": "Dedicated Swahili monolingual BERT model",
    },
    "afriberta": {
        "hf_id": "castorini/afriberta_large",
        "language": "african_multilingual",
        "description": "AfriBERTa large model tailored for low-resource African languages",
    },
    "bert": {
        "hf_id": "bert-base-uncased",
        "language": "english",
        "description": "Standard BERT baseline",
    },
}


def resolve_model_id(model_name_or_alias: str) -> str:
    """Resolve model alias to HuggingFace model ID."""
    alias = model_name_or_alias.lower().strip()
    if alias in MODEL_REGISTRY:
        return MODEL_REGISTRY[alias]["hf_id"]
    return model_name_or_alias


def load_model_and_tokenizer(
    model_name_or_alias: str,
    num_labels: int = 2,
    is_multilabel: bool = False,
    cache_dir: Optional[str] = None
) -> Tuple[Any, Any]:
    """
    Load HuggingFace model and tokenizer with appropriate task configuration.

    Parameters
    ----------
    model_name_or_alias : str
        Model identifier from MODEL_REGISTRY or HuggingFace Hub.
    num_labels : int, default=2
        Number of output classification classes.
    is_multilabel : bool, default=False
        Whether the problem type is multi-label classification.
    cache_dir : str, optional
        Custom cache directory for weights.

    Returns
    -------
    tokenizer, model
    """
    hf_id = resolve_model_id(model_name_or_alias)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, cache_dir=cache_dir)

    problem_type = "multi_label_classification" if is_multilabel else "single_label_classification"
    config = AutoConfig.from_pretrained(
        hf_id,
        num_labels=num_labels,
        problem_type=problem_type,
        cache_dir=cache_dir
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id,
        config=config,
        cache_dir=cache_dir
    )

    return tokenizer, model
