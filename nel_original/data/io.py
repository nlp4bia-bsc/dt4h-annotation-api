"""Tabular, ontology and text-collection input/output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..schemas import Concept, GazetteerEntry, HierarchyEdge, MentionAnnotation
from .brat import load_brat_ann
from .validators import parse_bool, require_columns


def load_annotations_tsv(path: str | Path, encoding: str = "utf-8") -> list[MentionAnnotation]:
    """Load the established mention-annotation TSV format."""
    dataframe = pd.read_csv(path, sep="\t", encoding=encoding)
    required = ["filename", "label", "start_span", "end_span", "text", "code"]
    require_columns(dataframe.columns.tolist(), required, "annotations")
    known_columns = {
        "filename",
        "label",
        "start_span",
        "end_span",
        "text",
        "code",
        "is_abbreviation",
        "is_composite",
        "is_compossite",
        "need_context",
    }

    annotations: list[MentionAnnotation] = []
    for row in dataframe.to_dict("records"):
        if pd.isna(row["code"]) or not str(row["code"]).strip():
            raise ValueError("Empty code in annotations row")
        composite_value = row.get("is_composite", row.get("is_compossite"))
        annotations.append(
            MentionAnnotation(
                filename=str(row["filename"]),
                label=None if pd.isna(row.get("label")) else str(row.get("label")),
                start_span=int(row["start_span"]) if pd.notna(row["start_span"]) else None,
                end_span=int(row["end_span"]) if pd.notna(row["end_span"]) else None,
                text=str(row["text"]),
                code=str(row["code"]),
                is_abbreviation=parse_bool(row.get("is_abbreviation")),
                is_composite=parse_bool(composite_value),
                need_context=parse_bool(row.get("need_context")),
                metadata={key: value for key, value in row.items() if key not in known_columns},
            )
        )
    return annotations


def load_gazetteer_tsv(path: str | Path, encoding: str = "utf-8") -> list[GazetteerEntry]:
    """Load gazetteer entries from a TSV containing at least `term` and `code`."""
    dataframe = pd.read_csv(path, sep="\t", encoding=encoding)
    require_columns(dataframe.columns.tolist(), ["term", "code"], "gazetteer")
    return [
        GazetteerEntry(
            term=str(row["term"]),
            code=str(row["code"]),
            label=None if pd.isna(row.get("label")) else row.get("label"),
            semantic_tag=row.get("semantic_tag", row.get("sem_tag")),
            metadata={
                key: value
                for key, value in row.items()
                if key not in {"term", "code", "label", "semantic_tag", "sem_tag"}
            },
        )
        for row in dataframe.to_dict("records")
    ]


def load_concepts_tsv(path: str | Path) -> list[Concept]:
    """Load ontology concepts from TSV."""
    dataframe = pd.read_csv(path, sep="\t")
    require_columns(dataframe.columns.tolist(), ["code", "term"], "concepts")
    concepts: list[Concept] = []
    for row in dataframe.to_dict("records"):
        alias_value = row.get("aliases")
        aliases = str(alias_value).split("|") if pd.notna(alias_value) and str(alias_value) else []
        concepts.append(
            Concept(
                code=str(row["code"]),
                term=str(row["term"]),
                label=None if pd.isna(row.get("label")) else row.get("label"),
                semantic_tag=row.get("semantic_tag", row.get("sem_tag")),
                aliases=aliases,
            )
        )
    return concepts


def load_hierarchy_tsv(path: str | Path) -> list[HierarchyEdge]:
    """Load directed parent-to-child ontology edges from TSV."""
    dataframe = pd.read_csv(path, sep="\t")
    require_columns(dataframe.columns.tolist(), ["parent_code", "child_code"], "hierarchy")
    return [
        HierarchyEdge(
            parent_code=str(row["parent_code"]),
            child_code=str(row["child_code"]),
            relation=None if pd.isna(row.get("relation")) else row.get("relation"),
        )
        for row in dataframe.to_dict("records")
    ]


def load_text_dir(path: str | Path, encoding: str = "utf-8") -> dict[str, str]:
    """Load all `.txt` files in a directory, keyed by filename stem."""
    return {text_path.stem: text_path.read_text(encoding=encoding) for text_path in sorted(Path(path).glob("*.txt"))}


def load_brat_dir(path: str | Path) -> list[MentionAnnotation]:
    """Load all BRAT `.ann` files in a directory."""
    annotations: list[MentionAnnotation] = []
    for annotation_path in sorted(Path(path).glob("*.ann")):
        annotations.extend(load_brat_ann(annotation_path))
    return annotations


def read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read TSV, JSONL or Parquet data based on the file extension."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    suffix = input_path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(input_path, sep="\t", **kwargs)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(input_path, lines=True, **kwargs)
    if suffix == ".parquet":
        try:
            return pd.read_parquet(input_path, **kwargs)
        except ImportError as exc:
            raise ImportError("Parquet support requires the 'parquet' extra.") from exc
    raise ValueError(f"Unsupported input format: {suffix}")


def write_table(
    dataframe: pd.DataFrame,
    path: str | Path,
    *,
    overwrite: bool = False,
    **kwargs: Any,
) -> Path:
    """Write a dataframe as TSV, JSONL or Parquet without accidental overwrite."""
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        serializable = dataframe.copy()
        for column in serializable.columns:
            if serializable[column].map(lambda value: isinstance(value, (list, dict, tuple))).any():
                serializable[column] = serializable[column].map(
                    lambda value: (
                        json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict, tuple)) else value
                    )
                )
        serializable.to_csv(output_path, sep="\t", index=False, **kwargs)
    elif suffix in {".jsonl", ".ndjson"}:
        dataframe.to_json(output_path, orient="records", lines=True, force_ascii=False, **kwargs)
    elif suffix == ".parquet":
        try:
            dataframe.to_parquet(output_path, index=False, **kwargs)
        except ImportError as exc:
            raise ImportError("Parquet support requires the 'parquet' extra.") from exc
    else:
        raise ValueError(f"Unsupported output format: {suffix}")
    return output_path
