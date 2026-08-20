"""BRAT annotation readers."""

from __future__ import annotations

from pathlib import Path

from ..schemas import MentionAnnotation


def load_brat_ann(path: str | Path) -> list[MentionAnnotation]:
    """Load text-bound BRAT annotations and normalization references."""
    annotation_path = Path(path)
    text_bound: dict[str, tuple[str, int, int, str]] = {}
    references: dict[str, str] = {}

    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("T"):
            annotation_id, span_data, mention = line.split("\t")
            label, start, end = span_data.split()[:3]
            text_bound[annotation_id] = (label, int(start), int(end), mention)
        elif line.startswith("N"):
            parts = line.split()
            if len(parts) >= 4:
                references[parts[2]] = parts[3]

    return [
        MentionAnnotation(
            filename=annotation_path.stem,
            label=label,
            start_span=start,
            end_span=end,
            text=mention,
            code=references.get(annotation_id),
        )
        for annotation_id, (label, start, end, mention) in text_bound.items()
    ]
