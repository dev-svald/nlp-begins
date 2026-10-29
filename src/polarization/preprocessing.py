"""
Text preprocessing and cleaning utilities for multilingual polarization detection.
"""

from __future__ import annotations

import re
from typing import Optional

try:
    import emoji
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False


def clean_text_strategy_1(text: str) -> str:
    """
    Cleaning Strategy 1:
    - Remove URLs and hyperlinks
    - Remove @user mentions
    - Convert emojis to descriptive text tokens (demojize)
    - Normalize whitespace
    """
    text = str(text)

    # 1. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # 2. Remove @user mentions
    text = re.sub(r"@\w+", "", text)

    # 3. Emoji to text conversion
    if HAS_EMOJI:
        text = emoji.demojize(text, delimiters=(" ", " "))
    else:
        # Fallback regex emoji substitution if emoji package not installed
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub(" ", text)

    # 4. Normalize spaces
    text = " ".join(text.split())
    return text


def clean_text_strategy_2(text: str) -> str:
    """
    Cleaning Strategy 2:
    - Remove URLs and hyperlinks
    - Remove @user mentions
    - Remove emojis entirely
    - Normalize whitespace
    """
    text = str(text)

    # 1. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # 2. Remove @user mentions
    text = re.sub(r"@\w+", "", text)

    # 3. Emoji removal
    if HAS_EMOJI and hasattr(emoji, "replace_emoji"):
        text = emoji.replace_emoji(text, replace="")
    else:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)

    # 4. Normalize spaces
    text = " ".join(text.split())
    return text


def clean_text(text: str, strategy: int = 1) -> str:
    """
    Dispatch text cleaning based on strategy index:
    - 0: Raw text (identity)
    - 1: Demojize + URL/user removal
    - 2: Delete emojis + URL/user removal
    """
    if strategy == 1:
        return clean_text_strategy_1(text)
    elif strategy == 2:
        return clean_text_strategy_2(text)
    return str(text)


def calculate_text_length(text: str) -> int:
    """Calculate token/word length of a text string."""
    return len(str(text).split())
