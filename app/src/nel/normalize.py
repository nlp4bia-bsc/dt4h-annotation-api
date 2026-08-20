"""Surface-form normalisation for lexical candidate generation.

Ported from the reference implementation's ``preprocessing.normalize_text``.

Scope note: this is applied by the **lexical** generators only.  It is
deliberately *not* applied to the dense path.  The NEL models in the registry
(SapBERT, ClinLinker-KB-P) were trained on cased clinical terms, so lowercasing
their input would change every embedding, invalidate every persisted FAISS
index, and shift retrieval quality in a direction nothing in this repository
can measure.  If that is ever tried, it must be measured, not assumed.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_text(
    text: str,
    lowercase: bool = True,
    strip_accents: bool = False,
    normalize_punct: bool = False,
) -> str:
    """Collapse whitespace and optionally case, accents and punctuation.

    Parameters
    ----------
    text:
        Input surface form.
    lowercase:
        Fold to lowercase.
    strip_accents:
        Drop Unicode combining marks, so ``"cáncer"`` matches ``"cancer"``.
        Off by default: in Spanish and Romanian clinical text accents can be
        the only difference between distinct words.
    normalize_punct:
        Replace punctuation with spaces.  Off by default because it destroys
        dosage and code-like forms such as ``"covid-19"``.

    Returns
    -------
    str
        The normalised text.
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
