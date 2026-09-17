"""Command-line interface for candidate retrieval and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .data import read_table
from .evaluation import evaluate_candidate_dataframe, load_snomed_graph_pickle
from .retrieval import CandidateRetrievalPipeline, HerbertFaissBiEncoder, build_vocabulary


def build_parser() -> argparse.ArgumentParser:
    """Build the package command-line parser."""
    parser = argparse.ArgumentParser(
        prog="nlp4bia-linking",
        description="Biomedical entity-linking candidate retrieval and evaluation.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("version", help="Print the installed version.")

    retrieve = subparsers.add_parser(
        "retrieve",
        help="Build a transformer FAISS index and retrieve candidates.",
    )
    retrieve.add_argument("--input", type=Path, required=True, help="TSV, JSONL or Parquet mention dataset.")
    retrieve.add_argument("--gazetteer", type=Path, required=True, help="Primary term-code vocabulary file.")
    retrieve.add_argument(
        "--vocabulary-source",
        action="append",
        type=Path,
        default=[],
        help="Additional vocabulary source, such as a training set. Repeat as needed.",
    )
    retrieve.add_argument("--model", required=True, help="Hugging Face model identifier or local path.")
    retrieve.add_argument("--output", type=Path, required=True, help="Prediction output: TSV, JSONL or Parquet.")
    retrieve.add_argument("--metrics-output", type=Path, help="Optional metrics JSON path.")
    retrieve.add_argument("--mention-column", default="text")
    retrieve.add_argument("--gold-column", default="code")
    retrieve.add_argument("--term-column", help="Vocabulary term column. Auto-detected when omitted.")
    retrieve.add_argument("--vocabulary-code-column", help="Vocabulary code column. Auto-detected when omitted.")
    retrieve.add_argument("--index-type", default="FlatIP", choices=HerbertFaissBiEncoder.supported_index_types())
    retrieve.add_argument("--max-length", type=int, default=256)
    retrieve.add_argument("--batch-size", type=int, default=64)
    retrieve.add_argument("--top-k", type=int, default=200)
    retrieve.add_argument("--k", type=int, nargs="+", default=[1, 5, 25, 100, 200])
    retrieve.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:<index>.")
    retrieve.add_argument("--pooling", choices=["mean", "attention_mask_mean", "cls"], default="mean")
    retrieve.add_argument("--overfetch-factor", type=int, default=5)
    retrieve.add_argument("--graph", type=Path, help="Optional trusted hierarchy Pickle for gold evaluation.")
    retrieve.add_argument("--cpu-faiss", action="store_true", help="Keep the FAISS index on CPU.")
    retrieve.add_argument("--overwrite", action="store_true")
    retrieve.add_argument("--verbose", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate ranked candidate codes.")
    evaluate.add_argument("predictions", type=Path)
    evaluate.add_argument("--gold-column", default="gold_code")
    evaluate.add_argument("--codes-column", default="codes")
    evaluate.add_argument("--k", type=int, nargs="+", default=[1, 5, 25, 100, 200])
    evaluate.add_argument("--graph", type=Path)
    evaluate.add_argument("--output", type=Path)
    return parser


def _derived_metrics_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.metrics.json")


def _run_retrieve(args: argparse.Namespace) -> int:
    vocabulary = build_vocabulary(
        args.gazetteer,
        args.vocabulary_source,
        term_column=args.term_column,
        code_column=args.vocabulary_code_column,
    )
    retriever = HerbertFaissBiEncoder(
        model_name_or_path=args.model,
        f_type=args.index_type,
        max_length=args.max_length,
        vocab=vocabulary,
        batch_size=args.batch_size,
        device=args.device,
        verbose=int(args.verbose),
        overfetch_factor=args.overfetch_factor,
        use_gpu=not args.cpu_faiss,
        pooling=args.pooling,
    )
    pipeline = CandidateRetrievalPipeline(
        retriever,
        top_k=args.top_k,
        k_values=args.k,
        graph_path=args.graph,
    )
    pipeline.fit(vocabulary, batch_size=args.batch_size)
    input_frame = read_table(args.input)
    has_gold = (
        args.gold_column in input_frame.columns
        and (input_frame[args.gold_column].notna() & input_frame[args.gold_column].astype(str).str.strip().ne("")).any()
    )
    metrics_path = args.metrics_output or (_derived_metrics_path(args.output) if has_gold else None)
    result = pipeline.run(
        input_frame,
        mention_column=args.mention_column,
        gold_column=args.gold_column,
        batch_size=args.batch_size,
        output_path=args.output,
        metrics_output_path=metrics_path,
        overwrite=args.overwrite,
    )
    summary = {
        "output": str(args.output),
        "rows": len(result.predictions),
        "vocabulary_pairs": len(vocabulary),
        "unique_codes": int(vocabulary["code"].nunique()),
        "torch_device": retriever.device,
        "faiss_device": retriever.resolved_faiss_device,
        "evaluated": result.evaluated,
    }
    if metrics_path is not None and result.metrics is not None:
        summary["metrics_output"] = str(metrics_path)
        summary["metrics"] = result.metrics
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Execute the command-line interface and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {None, "version"}:
        print(__version__)
        return 0
    if args.command == "retrieve":
        return _run_retrieve(args)
    if args.command == "evaluate":
        dataframe = read_table(args.predictions)
        graph = undirected = None
        if args.graph:
            graph, undirected = load_snomed_graph_pickle(args.graph)
        metrics = evaluate_candidate_dataframe(
            dataframe,
            args.k,
            gold_col=args.gold_column,
            codes_col=args.codes_column,
            graph=graph,
            undirected_graph=undirected,
        )
        rendered = json.dumps(metrics, indent=2, ensure_ascii=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
