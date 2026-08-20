from __future__ import annotations

import ast
import json
import pickle
import re
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd


def read_required_tsv(path: str | Path, sep: str = "\t", **kwargs: Any) -> pd.DataFrame:
    """Read a required TSV file with a clear error when it is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    # Deliberately silent in library code.
    return pd.read_csv(path, sep=sep, **kwargs)


def parse_list_cell(value: Any) -> list[Any]:
    """Parse a dataframe cell containing a list-like value.

    The notebooks and scripts often persist candidate lists as JSON strings in
    TSV files. This helper accepts real lists/tuples, JSON strings, Python
    literal strings and scalar values. Missing values become an empty list.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if pd.isna(value):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            return json.loads(value)
        except Exception:
            pass
        try:
            return ast.literal_eval(value)
        except Exception:
            pass
    return [value]


def normalize_retrieval_candidate_columns(
    df: pd.DataFrame,
    codes_col: str = "codes",
    candidates_col: str = "candidates",
    scores_col: str = "scores",
    term_col: str = "term",
    code_col: str = "code",
) -> pd.DataFrame:
    """Normalize retrieval outputs to canonical candidate/training columns.

    Older cached files may not contain ``scores`` because the canonical public
    output was simplified to ``codes`` and ``candidates``. Triplet generation can
    still run without scores, so this function fills missing scores with
    ``None`` values aligned to the candidate list. It also tolerates legacy score
    column names such as ``thresholds`` or ``scores_list`` when present.

    The helper also normalizes common retrieval aliases:
    ``text`` → ``term`` and ``gold_code`` → ``code``. This keeps cached outputs
    from notebook/script 01 reusable for cross-encoder triplet generation.
    """
    out = df.copy()
    rename_map = {}
    if term_col not in out.columns and "text" in out.columns:
        rename_map["text"] = term_col
    if code_col not in out.columns and "gold_code" in out.columns:
        rename_map["gold_code"] = code_col
    if codes_col not in out.columns:
        legacy_codes_col = next((col for col in ("codes_list", "candidate_codes") if col in out.columns), None)
        if legacy_codes_col is not None:
            rename_map[legacy_codes_col] = codes_col
    if candidates_col not in out.columns:
        legacy_candidates_col = next(
            (col for col in ("candidates_list", "candidate_list", "terms") if col in out.columns), None
        )
        if legacy_candidates_col is not None:
            rename_map[legacy_candidates_col] = candidates_col
    if rename_map:
        out = out.rename(columns=rename_map)

    missing = [col for col in (codes_col, candidates_col) if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required retrieval candidate columns: {missing}")

    out[codes_col] = out[codes_col].apply(parse_list_cell)
    out[candidates_col] = out[candidates_col].apply(parse_list_cell)

    if scores_col in out.columns:
        out[scores_col] = out[scores_col].apply(parse_list_cell)
    else:
        legacy_score_col = next((col for col in ("thresholds", "scores_list") if col in out.columns), None)
        if legacy_score_col is not None:
            out[scores_col] = out[legacy_score_col].apply(parse_list_cell)
        else:
            out[scores_col] = out[codes_col].apply(lambda codes: [None] * len(codes))

    def _align_scores(row: pd.Series) -> list[Any]:
        codes = row[codes_col]
        scores = row[scores_col]
        if len(scores) == len(codes):
            return scores
        if not scores:
            return [None] * len(codes)
        return list(scores[: len(codes)]) + [None] * max(0, len(codes) - len(scores))

    out[scores_col] = out.apply(_align_scores, axis=1)
    return out


def normalize_term_code_df(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Normalize SympTEMIST-like dataframes to unique non-empty ``term, code`` rows."""
    out = df.copy()
    if "term" not in out.columns and text_col in out.columns:
        out = out.rename(columns={text_col: "term"})
    if "term" not in out.columns or "code" not in out.columns:
        raise ValueError("Expected columns term/text and code")
    out = out[["term", "code"]].dropna().copy()
    out["term"] = out["term"].astype(str).apply(lambda x: re.sub(r"[«»]", "", x).strip())
    out["code"] = out["code"].astype(str)
    return out[out["term"].astype(bool)].drop_duplicates().reset_index(drop=True)


def load_snomed_graph_pickle(path: str | Path) -> nx.DiGraph:
    """Load a NetworkX SNOMED graph pickle as a directed graph."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SNOMED graph pickle not found: {path}")
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if isinstance(obj, nx.DiGraph):
        return obj
    if isinstance(obj, nx.Graph):
        return nx.DiGraph(obj)
    if isinstance(obj, tuple) and obj and isinstance(obj[0], nx.Graph):
        return obj[0] if isinstance(obj[0], nx.DiGraph) else nx.DiGraph(obj[0])
    if isinstance(obj, dict):
        for key in ("graph", "G", "digraph", "DiGraph"):
            graph = obj.get(key)
            if isinstance(graph, nx.DiGraph):
                return graph
            if isinstance(graph, nx.Graph):
                return nx.DiGraph(graph)
    raise TypeError(f"Unsupported graph pickle payload type: {type(obj)!r}")


def load_symptemist_train_and_gazetteer(
    train_path: str | Path, gazetteer_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load SympTEMIST train and gazetteer TSVs and return train, gazetteer and concat gazetteer."""
    train_df = normalize_term_code_df(read_required_tsv(train_path), text_col="text")
    gazetteer_df = normalize_term_code_df(read_required_tsv(gazetteer_path), text_col="term")
    gaz_train_df = (
        pd.concat([train_df[["code", "term"]], gazetteer_df[["code", "term"]]], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return train_df, gazetteer_df, gaz_train_df


def build_code_to_terms(df: pd.DataFrame, code_col: str = "code", term_col: str = "term") -> dict[str, list[str]]:
    """Build a code → unique terms mapping from a dataframe."""
    code_to_terms: dict[str, list[str]] = {}
    for row in df[[code_col, term_col]].dropna().itertuples(index=False):
        code = str(getattr(row, code_col)) if hasattr(row, code_col) else str(row[0])
        term = str(getattr(row, term_col)) if hasattr(row, term_col) else str(row[1])
        term = term.strip()
        if term:
            code_to_terms.setdefault(code, [])
            if term not in code_to_terms[code]:
                code_to_terms[code].append(term)
    return code_to_terms


def append_unique_term(mapping: dict[str, list[str]], code: str, term: Any) -> None:
    """Append one or many terms to a code → terms mapping."""
    if term is None:
        return
    if isinstance(term, (list, tuple, set)):
        for item in term:
            append_unique_term(mapping, code, item)
        return
    value = str(term).strip()
    if not value:
        return
    mapping.setdefault(str(code), [])
    if value not in mapping[str(code)]:
        mapping[str(code)].append(value)


def enrich_code_to_terms_from_graph(mapping: dict[str, list[str]], graph: nx.Graph | None) -> None:
    """Enrich a code → terms mapping with common term/synonym node attributes."""
    if graph is None:
        return
    term_attrs = ("term", "preferred_term", "pt", "fsn", "label", "name")
    list_attrs = ("aliases", "synonyms", "terms", "descriptions")
    for node, attrs in graph.nodes(data=True):
        code = str(node)
        for attr in term_attrs:
            append_unique_term(mapping, code, attrs.get(attr))
        for attr in list_attrs:
            append_unique_term(mapping, code, attrs.get(attr))
