"""Mention extraction helpers for plain-text collections."""

from __future__ import annotations

from pathlib import Path

from ..schemas import MentionAnnotation


def mentions_from_text_dictionary(
    text_dictionary: list[str],
    texts_dir: str | Path,
    label: str | None = None,
    code: str | None = None,
) -> list[MentionAnnotation]:
    """Find all case-insensitive dictionary occurrences in a directory of text files."""
    mentions: list[MentionAnnotation] = []
    terms = [str(term).strip() for term in text_dictionary if str(term).strip()]

    for path in Path(texts_dir).glob("*.txt"):
        content = path.read_text(encoding="utf-8")
        lowered_content = content.lower()
        for term in terms:
            lowered_term = term.lower()
            search_start = 0
            while True:
                start = lowered_content.find(lowered_term, search_start)
                if start == -1:
                    break
                end = start + len(term)
                mentions.append(
                    MentionAnnotation(
                        filename=path.stem,
                        label=label,
                        start_span=start,
                        end_span=end,
                        text=content[start:end],
                        code=code,
                    )
                )
                search_start = start + 1
    return mentions
