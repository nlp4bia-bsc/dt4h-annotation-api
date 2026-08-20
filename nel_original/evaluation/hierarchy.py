from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import pandas as pd

from ..data import load_concepts_tsv, load_hierarchy_tsv
from ..schemas import Concept, HierarchyEdge


def build_snomed_graph(
    concepts: Iterable[Concept] | None = None,
    hierarchy_edges: Iterable[HierarchyEdge] | None = None,
) -> tuple[nx.DiGraph, nx.Graph]:
    """Build directed and undirected SNOMED-like graphs from concepts and hierarchy edges.

    Edges are added as `parent_code -> child_code`, matching SNOMED `is_a` semantics.
    Node attributes include `term`, `label` and `semantic_tag` when concepts are supplied.
    """
    graph = nx.DiGraph()
    for concept in concepts or []:
        graph.add_node(
            str(concept.code),
            term=concept.term,
            label=concept.label,
            semantic_tag=concept.semantic_tag,
            aliases=concept.aliases,
        )
    for edge in hierarchy_edges or []:
        parent = str(edge.parent_code)
        child = str(edge.child_code)
        graph.add_node(parent)
        graph.add_node(child)
        graph.add_edge(parent, child, relation=edge.relation or "is_a")
    return graph, graph.to_undirected()


def load_snomed_graph(
    concepts_path: str | None = None, hierarchy_path: str | None = None
) -> tuple[nx.DiGraph, nx.Graph]:
    """Load concepts/hierarchy TSV files and build a SNOMED-like graph."""
    concepts = load_concepts_tsv(concepts_path) if concepts_path else []
    hierarchy = load_hierarchy_tsv(hierarchy_path) if hierarchy_path else []
    return build_snomed_graph(concepts, hierarchy)


def load_snomed_graph_pickle(path: str | Path) -> tuple[nx.DiGraph, nx.Graph]:
    """Load a pickled NetworkX SNOMED graph and return directed/undirected views.

    The expected object is a NetworkX graph with SNOMED `is_a` semantics encoded
    as parent→child edges. For convenience, this also accepts `(graph,
    undirected_graph)` tuples or dictionaries containing `graph`/`G` and `UG`
    keys, which are common when precomputing SNOMED artifacts.
    """
    with Path(path).open("rb") as handle:
        obj = pickle.load(handle)

    undirected_graph = None
    if isinstance(obj, tuple) and obj:
        graph = obj[0]
        if len(obj) > 1 and isinstance(obj[1], nx.Graph):
            undirected_graph = obj[1]
    elif isinstance(obj, dict):
        graph = None
        for key in ("graph", "G", "digraph"):
            if key in obj:
                graph = obj[key]
                break
        undirected_graph = None
        for key in ("UG", "undirected_graph"):
            if key in obj:
                undirected_graph = obj[key]
                break
    else:
        graph = obj

    if not isinstance(graph, nx.Graph):
        raise TypeError("Pickle must contain a NetworkX graph, a (graph, UG) tuple or a graph dictionary")
    directed_graph = graph if isinstance(graph, nx.DiGraph) else nx.DiGraph(graph)
    if undirected_graph is None:
        undirected_graph = directed_graph.to_undirected()
    return directed_graph, undirected_graph


def snomed_graph_distance_and_direction(
    graph: nx.DiGraph,
    undirected_graph: nx.Graph,
    gold_code: str | int | None,
    pred_code: str | int | None,
    return_path: bool = False,
) -> dict[str, Any]:
    """Classify top-1 prediction relative to gold as exact/narrow/broad/unrelated.

    Direction conventions assume a parent→child directed graph:
    - `exact`: same code.
    - `narrow`: predicted code is a descendant/specialization of the gold code.
    - `broad`: predicted code is an ancestor/generalization of the gold code, or a same-tag sibling.
    - `unrelated`: missing/disconnected/different semantic-tag relationship.
    """
    gold = str(gold_code) if gold_code is not None else None
    pred = str(pred_code) if pred_code is not None else None

    gold_tag = graph.nodes[gold].get("semantic_tag") if gold in graph else None
    pred_tag = graph.nodes[pred].get("semantic_tag") if pred in graph else None
    same_tag = bool(gold_tag and pred_tag and gold_tag == pred_tag)

    out: dict[str, Any] = {
        "gold": gold,
        "pred": pred,
        "distance": None,
        "direction": None,
        "gold_semantic_tag": gold_tag,
        "pred_semantic_tag": pred_tag,
        "same_semantic_tag": same_tag,
    }
    if return_path:
        out["path"] = None

    if gold is None or pred is None:
        out["direction"] = "unrelated"
        return out

    if gold == pred:
        out["distance"] = 0
        out["direction"] = "exact"
        if return_path:
            out["path"] = [gold]
        return out

    if gold not in graph or pred not in graph:
        out["direction"] = "unrelated"
        return out

    if nx.has_path(graph, gold, pred):
        path = nx.shortest_path(graph, gold, pred)
        out["distance"] = len(path) - 1
        out["direction"] = "narrow"
        if return_path:
            out["path"] = path
        return out

    if nx.has_path(graph, pred, gold):
        path = nx.shortest_path(graph, pred, gold)
        out["distance"] = len(path) - 1
        out["direction"] = "broad"
        if return_path:
            out["path"] = path
        return out

    try:
        undirected_path = nx.shortest_path(undirected_graph, gold, pred)
        out["distance"] = len(undirected_path) - 1
        if return_path:
            out["path"] = undirected_path
    except nx.NetworkXNoPath:
        out["direction"] = "unrelated"
        return out

    if not same_tag:
        out["direction"] = "unrelated"
        return out

    ancestors_gold = {gold} | nx.ancestors(graph, gold)
    ancestors_pred = {pred} | nx.ancestors(graph, pred)
    common = ancestors_gold & ancestors_pred
    if not common:
        out["direction"] = "unrelated"
        return out

    best_ancestor = None
    best_score = None
    best_gold_distance = None
    for ancestor in common:
        try:
            distance_to_gold = nx.shortest_path_length(graph, ancestor, gold)
            distance_to_pred = nx.shortest_path_length(graph, ancestor, pred)
        except nx.NetworkXNoPath:
            continue
        score = distance_to_gold + distance_to_pred
        if best_score is None or score < best_score:
            best_score = score
            best_ancestor = ancestor
            best_gold_distance = distance_to_gold

    if best_ancestor is None or best_gold_distance is None:
        out["direction"] = "unrelated"
        return out

    if best_ancestor != gold and best_gold_distance > 0:
        out["direction"] = "broad"
        return out

    if nx.has_path(graph, gold, best_ancestor):
        out["direction"] = "narrow"
        return out

    out["direction"] = "broad"
    return out


def _parse_codes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            return [str(item) for item in json.loads(text)]
        return [text]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def first_code(value: Any) -> str | None:
    """Return the first candidate code from list-like, JSON-list or scalar values."""
    codes = _parse_codes(value)
    return codes[0] if codes else None


def calculate_recall_at_k(
    df: pd.DataFrame, k_values: list[int], gold_col: str = "code", codes_col: str = "codes"
) -> dict[int, float]:
    """Calculate Recall@k over unique predicted codes stored as lists or JSON lists."""
    recalls = {k: 0 for k in k_values}
    total_rows = len(df)
    if total_rows == 0:
        return {k: 0.0 for k in k_values}

    for _, row in df.iterrows():
        true_code = str(row[gold_col])
        seen: set[str] = set()
        unique_candidates = [code for code in _parse_codes(row[codes_col]) if not (code in seen or seen.add(code))]
        for k in k_values:
            if true_code in unique_candidates[:k]:
                recalls[k] += 1
    return {k: recalls[k] / total_rows for k in k_values}


def graph_metrics_for_dataframe(
    df: pd.DataFrame,
    graph: nx.DiGraph,
    undirected_graph: nx.Graph,
    gold_col: str = "code",
    codes_col: str = "codes",
    label_col: str | None = "label",
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """Compute SNOMED graph top-1 metrics and Recall@k globally and by label."""
    k_values = k_values or [1, 5]
    metrics_df = df.apply(
        lambda row: snomed_graph_distance_and_direction(
            graph,
            undirected_graph,
            gold_code=row[gold_col],
            pred_code=first_code(row[codes_col]),
        ),
        axis=1,
        result_type="expand",
    )

    direction_summary = (
        metrics_df.groupby("direction", dropna=False)
        .agg(count=("direction", "size"), mean_distance=("distance", "mean"))
        .reset_index()
    )
    direction_summary["normalized"] = direction_summary["count"] / len(metrics_df) if len(metrics_df) else 0.0

    report: dict[str, Any] = {
        "direction_summary": direction_summary.to_dict("records"),
        "same_semantic_tag_summary": metrics_df.groupby("same_semantic_tag", dropna=False)
        .agg(count=("same_semantic_tag", "size"), mean_distance=("distance", "mean"))
        .reset_index()
        .to_dict("records"),
        "mean_distance_global": float(metrics_df["distance"].mean()) if len(metrics_df) else 0.0,
        "recall_at_k": calculate_recall_at_k(df, k_values, gold_col=gold_col, codes_col=codes_col),
        "per_row": metrics_df,
    }

    if label_col and label_col in df.columns:
        by_label = {}
        for label, label_df in df.groupby(label_col):
            label_metrics = metrics_df.loc[label_df.index]
            by_label[str(label)] = {
                "n": len(label_df),
                "direction_summary": label_metrics.groupby("direction", dropna=False)
                .agg(count=("direction", "size"), mean_distance=("distance", "mean"))
                .reset_index()
                .to_dict("records"),
                "recall_at_k": calculate_recall_at_k(label_df, k_values, gold_col=gold_col, codes_col=codes_col),
                "mean_distance": float(label_metrics["distance"].mean()) if len(label_metrics) else 0.0,
            }
        report["by_label"] = by_label
    return report


def load_ontology_graph_pickle(path: str | Path) -> tuple[nx.DiGraph, nx.Graph]:
    """Compatibility alias for :func:`load_snomed_graph_pickle`."""
    return load_snomed_graph_pickle(path)


def ontology_distance_and_direction(
    graph: nx.DiGraph,
    undirected_graph: nx.Graph,
    gold_code: str | int | None,
    pred_code: str | int | None,
    return_path: bool = False,
) -> dict[str, Any]:
    """Compatibility view using ``non_rel`` for unrelated predictions."""
    result = snomed_graph_distance_and_direction(
        graph,
        undirected_graph,
        gold_code,
        pred_code,
        return_path=return_path,
    )
    if result.get("direction") == "unrelated":
        result["direction"] = "non_rel"
    return result


def ontology_summary_from_codes(
    gold_codes: Iterable[Any],
    predicted_codes: Iterable[Any],
    graph: nx.DiGraph,
    undirected_graph: nx.Graph,
) -> dict[str, float | int]:
    """Aggregate exact, narrow, broad and non-related top-1 graph results."""
    from .metrics import parse_code_list, unique_preserve_order

    directions = {"exact": 0, "narrow": 0, "broad": 0, "non_rel": 0}
    distances: list[float] = []
    gold_list = list(gold_codes)
    predicted_list = list(predicted_codes)
    for gold, predictions in zip(gold_list, predicted_list):
        unique = unique_preserve_order(parse_code_list(predictions))
        prediction = unique[0] if unique else None
        result = ontology_distance_and_direction(graph, undirected_graph, gold, prediction)
        direction = str(result.get("direction") or "non_rel")
        directions[direction if direction in directions else "non_rel"] += 1
        if result.get("distance") is not None:
            distances.append(float(result["distance"]))
    n = len(gold_list)
    output: dict[str, float | int] = {}
    for direction, count in directions.items():
        output[f"{direction}_count"] = count
        output[f"{direction}_rate"] = count / n if n else 0.0
    output["mean_graph_distance"] = sum(distances) / len(distances) if distances else 0.0
    return output
