"""Text normalization utilities."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(
    text: str,
    lowercase: bool = True,
    strip_accents: bool = False,
    normalize_punct: bool = False,
) -> str:
    """Normalize whitespace, case, accents and punctuation.

    Args:
        text: Input text.
        lowercase: Convert text to lowercase.
        strip_accents: Remove Unicode combining accent marks.
        normalize_punct: Replace punctuation with spaces.

    Returns:
        The normalized text.
    """
    normalized = " ".join(str(text).split())
    if lowercase:
        normalized = normalized.lower()
    if strip_accents:
        normalized = "".join(
            character
            for character in unicodedata.normalize("NFD", normalized)
            if unicodedata.category(character) != "Mn"
        )
    if normalize_punct:
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        normalized = " ".join(normalized.split())
    return normalized
