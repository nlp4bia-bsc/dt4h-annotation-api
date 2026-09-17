from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..evaluation.metrics import retrieval_metrics_from_codes
from ..preprocessing import normalize_text
from ..utils.device import optional_module, resolve_faiss_device
from ..utils.profiling import Profiler


@dataclass(frozen=True)
class FaissExperimentConfig:
    """Configuration for one FAISS candidate-retrieval profiling run."""

    name: str
    index_factory: str = "Flat"
    metric: str = "ip"
    normalize: bool = True
    use_gpu: bool = True
    nprobe: int | None = None
    overfetch_factor: int = 4


def default_faiss_experiment_configs() -> list[FaissExperimentConfig]:
    """Return GPU-first FAISS configurations covering flat, normalized and quantized variants."""
    return [
        FaissExperimentConfig(name="flat_ip_normalized", index_factory="Flat", metric="ip", normalize=True),
        FaissExperimentConfig(name="flat_ip_raw", index_factory="Flat", metric="ip", normalize=False),
        FaissExperimentConfig(name="flat_l2", index_factory="Flat", metric="l2", normalize=False),
        FaissExperimentConfig(name="hnsw32_ip_normalized", index_factory="HNSW32", metric="ip", normalize=True),
        FaissExperimentConfig(name="sq8_ip_normalized", index_factory="SQ8", metric="ip", normalize=True),
        FaissExperimentConfig(
            name="ivf32_flat_ip_normalized", index_factory="IVF32,Flat", metric="ip", normalize=True, nprobe=8
        ),
        FaissExperimentConfig(
            name="ivf32_sq8_ip_normalized", index_factory="IVF32,SQ8", metric="ip", normalize=True, nprobe=8
        ),
    ]


def prepare_faiss_training_index(train_df: pd.DataFrame, gazetteer_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate train annotations and gazetteer terms into a unique FAISS index table."""
    _require_columns(train_df, ["term", "code"], "train_df")
    _require_columns(gazetteer_df, ["term", "code"], "gazetteer_df")
    index_df = pd.concat([train_df[["term", "code"]], gazetteer_df[["term", "code"]]], ignore_index=True)
    index_df = index_df.dropna(subset=["term", "code"]).copy()
    index_df["term"] = index_df["term"].astype(str)
    index_df["code"] = index_df["code"].astype(str)
    return index_df.drop_duplicates(subset=["term", "code"]).reset_index(drop=True)


def evaluate_recall_at_k_from_codes(
    gold_codes: list[str], predicted_codes: list[list[str]], k_values: list[int]
) -> dict[int, float]:
    """Compute Recall@k with per-query unique predicted codes via the central metrics package."""
    metrics = retrieval_metrics_from_codes(gold_codes, predicted_codes, k_values)
    return {k: float(metrics[f"recall@{k}"]) for k in k_values}


def profile_faiss_biencoder_configs(
    train_df: pd.DataFrame,
    gold_standard_df: pd.DataFrame,
    gazetteer_df: pd.DataFrame,
    configs: list[FaissExperimentConfig] | None = None,
    k_values: list[int] | None = None,
    device: str = "auto",
    output_dir: str | Path | None = None,
    analyzer: str = "char",
    ngram_range: tuple[int, int] = (3, 5),
    profile: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Profile multiple FAISS retrieval configurations and compute Recall@k.

    Args:
        train_df: DataFrame with `term` and `code` columns used as extra index entries.
        gold_standard_df: DataFrame with query `term` and gold `code` columns.
        gazetteer_df: DataFrame with gazetteer `term` and `code` columns.
        configs: FAISS index configurations to evaluate.
        k_values: Recall cutoffs, defaults to [1, 5, 25, 100, 200].
        device: `auto`, `cuda`, `cuda:0` or `cpu`. GPU is used when available.
        output_dir: Optional directory where metrics, predictions and profile JSON are saved.
    """
    _require_columns(gold_standard_df, ["term", "code"], "gold_standard_df")
    configs = configs or default_faiss_experiment_configs()
    k_values = k_values or [1, 5, 25, 100, 200]
    max_k = max(k_values)
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    profiler = Profiler(enabled=profile, device=device)
    with profiler.track("prepare_index_and_queries", n_items=len(train_df) + len(gazetteer_df)):
        index_df = prepare_faiss_training_index(train_df, gazetteer_df)
        query_terms = gold_standard_df["term"].dropna().astype(str).tolist()
        gold_codes = gold_standard_df.loc[gold_standard_df["term"].notna(), "code"].astype(str).tolist()
        vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
        index_vectors = (
            vectorizer.fit_transform([normalize_text(term) for term in index_df["term"]]).astype(np.float32).toarray()
        )
        query_vectors = (
            vectorizer.transform([normalize_text(term) for term in query_terms]).astype(np.float32).toarray()
        )

    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for config in configs:
        try:
            predictions = _run_single_faiss_config(
                config=config,
                index_df=index_df,
                index_vectors=index_vectors,
                query_vectors=query_vectors,
                query_terms=query_terms,
                gold_codes=gold_codes,
                max_k=max_k,
                device=device,
                profiler=profiler,
            )
            recall = evaluate_recall_at_k_from_codes(gold_codes, [row["codes"] for row in predictions], k_values)
            metrics_rows.append(
                {
                    **asdict(config),
                    "status": "ok",
                    "n_index_rows": len(index_df),
                    "n_queries": len(query_terms),
                    **{f"recall@{k}": value for k, value in recall.items()},
                }
            )
            prediction_rows.extend({**row, "config": config.name} for row in predictions)
        except Exception as exc:  # noqa: BLE001 - notebook should continue profiling remaining configs.
            metrics_rows.append({**asdict(config), "status": "error", "error": str(exc)})

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    report = profiler.final_report()
    if output_path is not None:
        metrics_df.to_csv(output_path / "faiss_config_metrics.tsv", sep="\t", index=False)
        predictions_df.to_csv(output_path / "faiss_config_predictions.tsv", sep="\t", index=False)
        (output_path / "faiss_config_profile.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return metrics_df, predictions_df, report


def _run_single_faiss_config(
    config: FaissExperimentConfig,
    index_df: pd.DataFrame,
    index_vectors: np.ndarray,
    query_vectors: np.ndarray,
    query_terms: list[str],
    gold_codes: list[str],
    max_k: int,
    device: str,
    profiler: Profiler,
) -> list[dict[str, Any]]:
    faiss = optional_module("faiss")
    if faiss is None:
        raise ImportError("faiss-gpu or faiss-cpu is required to profile FAISS configurations")
    x = np.array(index_vectors, dtype=np.float32, copy=True)
    q = np.array(query_vectors, dtype=np.float32, copy=True)
    if config.normalize:
        faiss.normalize_L2(x)
        faiss.normalize_L2(q)
    metric = faiss.METRIC_INNER_PRODUCT if config.metric == "ip" else faiss.METRIC_L2
    with profiler.track(f"faiss.{config.name}.build", n_items=len(index_df), **asdict(config)):
        index = faiss.index_factory(x.shape[1], config.index_factory, metric)
        if not index.is_trained:
            index.train(x)
        index.add(x)
        resolved_device = resolve_faiss_device(device) if config.use_gpu else "cpu"
        if resolved_device.startswith("cuda") and hasattr(faiss, "StandardGpuResources"):
            gpu_id = int(resolved_device.split(":", 1)[1]) if ":" in resolved_device else 0
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, gpu_id, index)
        if config.nprobe is not None and hasattr(index, "nprobe"):
            index.nprobe = config.nprobe
    overfetch = min(len(index_df), max(max_k * config.overfetch_factor, max_k))
    with profiler.track(f"faiss.{config.name}.search", n_items=len(query_terms), max_k=max_k, overfetch=overfetch):
        distances, indices = index.search(q, overfetch)
    predictions = []
    for row_idx, (term, gold, row_distances, row_indices) in enumerate(
        zip(query_terms, gold_codes, distances, indices)
    ):
        codes: list[str] = []
        candidates: list[str] = []
        seen: set[str] = set()
        for score, idx in zip(row_distances, row_indices):
            if idx < 0:
                continue
            code = str(index_df.iloc[int(idx)]["code"])
            if code in seen:
                continue
            seen.add(code)
            codes.append(code)
            candidates.append(str(index_df.iloc[int(idx)]["term"]))
            if len(codes) >= max_k:
                break
        predictions.append(
            {
                "query_id": row_idx,
                "term": term,
                "gold_code": str(gold),
                "codes": codes,
                "candidates": candidates,
            }
        )
    return predictions


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
