from __future__ import annotations

import ast
import json
from collections.abc import Iterable, Sequence
from typing import Any


def parse_code_list(value: Any) -> list[str]:
    """Parse candidate-code lists from Python lists, JSON strings or scalar values."""
    if value is None:
        return []
    if isinstance(value, float):
        try:
            import math

            if math.isnan(value):
                return []
        except Exception:
            pass
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = [text]
        return parse_code_list(parsed)
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None and str(item) != ""]
    return [str(value)]


def unique_preserve_order(values: Iterable[Any]) -> list[str]:
    """Return unique string values while preserving their first occurrence order."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def retrieval_metrics_from_codes(
    gold_codes: Sequence[Any],
    predicted_codes: Sequence[Sequence[Any] | Any],
    k_values: Sequence[int],
) -> dict[str, float | int]:
    """Compute recall@k, MRR, coverage and mean candidates.

    Candidate lists are deduplicated by code before metrics are computed.
    """
    n = len(gold_codes)
    metrics: dict[str, float | int] = {"n_mentions": n}
    for k in k_values:
        metrics[f"recall@{k}"] = 0.0
    metrics.update({"mrr": 0.0, "coverage": 0.0, "mean_candidates": 0.0})
    if n == 0:
        return metrics

    normalized_predictions = [unique_preserve_order(parse_code_list(codes)) for codes in predicted_codes]
    normalized_gold = [str(code) for code in gold_codes]

    hits_at_k = {int(k): 0 for k in k_values}
    reciprocal_rank = 0.0
    covered = 0
    total_candidates = 0

    for gold, preds in zip(normalized_gold, normalized_predictions):
        total_candidates += len(preds)
        covered += bool(preds)
        if gold in preds:
            reciprocal_rank += 1.0 / (preds.index(gold) + 1)
        for k in k_values:
            hits_at_k[int(k)] += gold in preds[: int(k)]

    for k in k_values:
        metrics[f"recall@{int(k)}"] = hits_at_k[int(k)] / n
    metrics["mrr"] = reciprocal_rank / n
    metrics["coverage"] = covered / n
    metrics["mean_candidates"] = total_candidates / n
    return metrics


def evaluate_candidate_rows(
    rows: Sequence[dict[str, Any]],
    k_values: Sequence[int],
    gold_col: str = "gold_code",
    codes_col: str = "codes",
    method: str | None = None,
    graph: Any | None = None,
    undirected_graph: Any | None = None,
) -> dict[str, float | int | str]:
    """Evaluate serialized prediction rows produced by retrieval/reranking notebooks."""
    gold = [row.get(gold_col) for row in rows]
    predicted = [row.get(codes_col, []) for row in rows]
    metrics = retrieval_metrics_from_codes(gold, predicted, k_values)
    if method is not None:
        metrics = {"method": method, **metrics}
    if graph is not None and undirected_graph is not None:
        from .hierarchy import ontology_summary_from_codes

        graph_metrics = ontology_summary_from_codes(gold, predicted, graph, undirected_graph)
        metrics.update(graph_metrics)
    return metrics


def evaluate_candidate_dataframe(
    df: Any,
    k_values: Sequence[int],
    gold_col: str = "gold_code",
    codes_col: str = "codes",
    method: str | None = None,
    graph: Any | None = None,
    undirected_graph: Any | None = None,
) -> dict[str, float | int | str]:
    """Evaluate a pandas DataFrame with gold codes and candidate-code lists."""
    gold = df[gold_col].astype(str).tolist() if len(df) else []
    predicted = df[codes_col].tolist() if len(df) else []
    metrics = retrieval_metrics_from_codes(gold, predicted, k_values)
    if method is not None:
        metrics = {"method": method, **metrics}
    if graph is not None and undirected_graph is not None:
        from .hierarchy import ontology_summary_from_codes

        metrics.update(ontology_summary_from_codes(gold, predicted, graph, undirected_graph))
    return metrics


def evaluate_predictions(
    gold: list[str],
    predicted_ranked: list[list[str]],
    k: int = 5,
) -> dict[str, float | int]:
    """Compatibility wrapper returning Recall@k and MRR for ranked code lists."""
    return retrieval_metrics_from_codes(gold, predicted_ranked, [k])
