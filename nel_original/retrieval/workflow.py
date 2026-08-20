"""High-level candidate retrieval over tabular mention datasets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..data import read_table, write_table
from ..evaluation import evaluate_candidate_dataframe, load_snomed_graph_pickle

TableSource = pd.DataFrame | str | Path


@dataclass(slots=True)
class RetrievalResult:
    """Predictions and optional evaluation metrics from a retrieval run."""

    predictions: pd.DataFrame
    metrics: dict[str, Any] | None

    @property
    def evaluated(self) -> bool:
        """Return whether at least one valid gold code was evaluated."""
        return self.metrics is not None


def _read_source(source: TableSource) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return read_table(source)


def _resolve_column(
    dataframe: pd.DataFrame,
    explicit: str | None,
    candidates: Sequence[str],
    role: str,
) -> str:
    if explicit is not None:
        if explicit not in dataframe.columns:
            raise ValueError(f"Configured {role} column '{explicit}' is not present")
        return explicit
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    raise ValueError(f"Could not infer the {role} column. Tried: {', '.join(candidates)}")


def build_vocabulary(
    gazetteer: TableSource,
    additional_sources: Sequence[TableSource] | TableSource | None = None,
    *,
    term_column: str | None = None,
    code_column: str | None = None,
    deduplicate: bool = True,
    remove_guillemets: bool = True,
) -> pd.DataFrame:
    """Build a normalized term-code vocabulary from a gazetteer and optional data.

    Each source may use ``term``, ``text`` or ``mention`` for the surface form,
    and ``code``, ``gold_code`` or ``concept_id`` for the concept identifier.
    Explicit column names can be supplied when automatic detection is not
    appropriate.
    """

    if additional_sources is None:
        extra_sources: list[TableSource] = []
    elif isinstance(additional_sources, (pd.DataFrame, str, Path)):
        extra_sources = [additional_sources]
    else:
        extra_sources = list(additional_sources)
    sources = [*extra_sources, gazetteer]
    frames: list[pd.DataFrame] = []
    for source in sources:
        dataframe = _read_source(source)
        resolved_term = _resolve_column(
            dataframe,
            term_column,
            ("term", "text", "mention"),
            "vocabulary term",
        )
        resolved_code = _resolve_column(
            dataframe,
            code_column,
            ("code", "gold_code", "concept_id"),
            "vocabulary code",
        )
        frame = dataframe[[resolved_term, resolved_code]].dropna().copy()
        frame.columns = ["term", "code"]
        frame["term"] = frame["term"].astype(str)
        frame["code"] = frame["code"].astype(str)
        frame = frame[(frame["term"].str.strip() != "") & (frame["code"].str.strip() != "")]
        if remove_guillemets:
            frame["term"] = frame["term"].map(lambda value: re.sub(r"[«»]", "", value))
        frames.append(frame)

    vocabulary = pd.concat(frames, ignore_index=True)
    if vocabulary.empty:
        raise ValueError("Vocabulary contains no valid term-code pairs")
    if deduplicate:
        vocabulary = vocabulary.drop_duplicates(subset=["code", "term"], keep="first")
    return vocabulary.reset_index(drop=True)


class CandidateRetrievalPipeline:
    """Index a vocabulary, retrieve candidates and evaluate when gold exists.

    The pipeline preserves every input column and appends aligned ``codes``,
    ``candidates`` and ``scores`` lists. Evaluation is automatic only when the
    configured gold column exists and contains at least one non-empty value.
    """

    def __init__(
        self,
        retriever: Any,
        *,
        top_k: int = 200,
        k_values: Sequence[int] = (1, 5, 25, 100, 200),
        graph_path: str | Path | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.retriever = retriever
        self.top_k = int(top_k)
        self.k_values = tuple(int(value) for value in k_values)
        self.graph_path = Path(graph_path) if graph_path is not None else None
        self._graph: Any | None = None
        self._undirected_graph: Any | None = None

    def fit(self, vocabulary: pd.DataFrame, *, batch_size: int | None = None) -> "CandidateRetrievalPipeline":
        """Set the vocabulary and build the retriever index."""
        if hasattr(self.retriever, "fit_faiss"):
            self.retriever.fit_faiss(vocab=vocabulary, batch_size=batch_size)
        elif hasattr(self.retriever, "build_index"):
            self.retriever.build_index(vocabulary)
        else:
            raise TypeError("Retriever must implement fit_faiss or build_index")
        return self

    def _load_graph(self) -> tuple[Any | None, Any | None]:
        if self.graph_path is None:
            return None, None
        if self._graph is None or self._undirected_graph is None:
            self._graph, self._undirected_graph = load_snomed_graph_pickle(self.graph_path)
        return self._graph, self._undirected_graph

    def run(
        self,
        data: TableSource,
        *,
        mention_column: str = "text",
        gold_column: str | None = "code",
        batch_size: int | None = None,
        output_path: str | Path | None = None,
        metrics_output_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> RetrievalResult:
        """Retrieve candidates and evaluate only rows with valid gold codes."""
        dataframe = _read_source(data)
        if dataframe.empty:
            raise ValueError("Input data contains no rows")
        if output_path is not None and Path(output_path).exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {Path(output_path)}")
        if mention_column not in dataframe.columns:
            raise ValueError(f"Mention column '{mention_column}' is not present")

        texts = dataframe[mention_column].fillna("").astype(str).tolist()
        if not hasattr(self.retriever, "get_candidates"):
            raise TypeError("Retriever must implement get_candidates(texts, k, batch_size)")
        terms, codes, scores = self.retriever.get_candidates(
            texts,
            k=self.top_k,
            batch_size=batch_size,
        )

        predictions = dataframe.copy()
        predictions["codes"] = codes
        predictions["candidates"] = terms
        predictions["scores"] = scores
        predictions["predicted_code"] = [items[0] if items else None for items in codes]
        predictions["predicted_term"] = [items[0] if items else None for items in terms]
        predictions["predicted_score"] = [items[0] if items else None for items in scores]
        predictions["retrieval_method"] = getattr(
            self.retriever,
            "method",
            self.retriever.__class__.__name__,
        )

        metrics: dict[str, Any] | None = None
        if gold_column is not None and gold_column in predictions.columns:
            gold = predictions[gold_column]
            valid_gold = gold.notna() & gold.astype(str).str.strip().ne("")
            if valid_gold.any():
                evaluation_frame = predictions.loc[valid_gold, [gold_column, "codes"]].copy()
                graph, undirected = self._load_graph()
                metrics = evaluate_candidate_dataframe(
                    evaluation_frame,
                    self.k_values,
                    gold_col=gold_column,
                    codes_col="codes",
                    method=str(predictions["retrieval_method"].iloc[0]) if len(predictions) else None,
                    graph=graph,
                    undirected_graph=undirected,
                )
                metrics["n_input_rows"] = int(len(predictions))
                metrics["n_evaluated_rows"] = int(valid_gold.sum())
                metrics["n_rows_without_gold"] = int((~valid_gold).sum())

        if output_path is not None:
            write_table(predictions, output_path, overwrite=overwrite)
        if metrics is not None and metrics_output_path is not None:
            metrics_path = Path(metrics_output_path)
            if metrics_path.exists() and not overwrite:
                raise FileExistsError(f"Output file already exists: {metrics_path}")
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        return RetrievalResult(predictions=predictions, metrics=metrics)
