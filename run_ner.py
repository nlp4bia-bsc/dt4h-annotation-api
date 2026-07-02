"""
run_ner.py — NER-only batch inference over data/{lang}/ samples.

Usage:
    uv run run_ner.py

For each language directory found in data/:
  1. Checks the registry: all NER models must have a local_path (i.e. downloaded).
  2. Reads all .txt files from data/{lang}/.
  3. Runs NER inference across all registered entity types (disease, symptom, etc.).
  4. Writes results/{lang}/raw/{stem}.ann   — flat entity spans per document.
  5. Writes results/{lang}/formatted/{stem}.json — DT4H CDM v2 JSON per document.
  6. Writes results/{lang}/{lang}.tsv — all annotations across all files, sorted by filename then start span.
  7. Releases model memory before processing the next language.
"""

import argparse
import csv
import gc
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
# tokenization false alarm since preprococessing handles model size overflow
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

def _check_registry(resolver, lang: str, entity_filter: list[str] | None = None) -> list[Path] | None:
    """
    Return list of local NER model paths for *lang*, or None if any are missing.

    Reads local_path from the registry YAML directly: a null entry means the
    model has not been downloaded yet, so the whole language is skipped.
    If *entity_filter* is given, only those entity types are checked/used.
    """
    from app.model_manager.resolver import ModelNotFoundError

    ner_cfg = resolver.registry.get("ner", {}).get(lang, {})
    registered = [e for e in ner_cfg if e != "negation"]
    entity_types = [e for e in registered if e in entity_filter] if entity_filter else registered

    if not entity_types:
        log.warning("[%s] No entity types registered — skip.", lang)
        return None

    model_paths: list[Path] = []
    for entity in entity_types:
        if ner_cfg[entity].get("local_path") is None:
            log.warning("[%s] '%s' model not downloaded (local_path is null) — skip.", lang, entity)
            return None
        try:
            path, repo_id = resolver.get_ner_path(lang, entity)
        except (ModelNotFoundError, FileNotFoundError) as exc:
            log.warning("[%s] '%s' model unavailable: %s — skip.", lang, entity, exc)
            return None
        if repo_id is not None:
            log.warning("[%s] '%s' model needs download — skip.", lang, entity)
            return None
        model_paths.append(path)

    return model_paths


def _write_ann(path: Path, rows: list[tuple[str, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: (r[0], r[1]["start"]))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["filename", "ner_class", "start", "end", "span", "ner_score"])
        for filename, ann in sorted_rows:
            writer.writerow([filename, ann["ner_class"], ann["start"], ann["end"], ann["span"], ann["ner_score"]])


def _write_tsv(path: Path, annotations: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["ner_class", "start", "end", "span", "ner_score"],
            delimiter="\t",
        )
        writer.writeheader()
        for ann in annotations:
            writer.writerow({
                "ner_class": ann["ner_class"],
                "start":     ann["start"],
                "end":       ann["end"],
                "span":      ann["span"],
                "ner_score": ann["ner_score"],
            })


def _write_json(path: Path, text: str, annotations: list[dict], formatter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Patch NER-only annotations: Dt4hFormatter expects NEL fields; all are Optional
    patched = []
    for ann in annotations:
        a = dict(ann)
        a.setdefault("term", None)
        a.setdefault("code", None)
        a.setdefault("nel_score", None)
        patched.append(a)
    result = formatter.serialize(text, patched, {})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NER-only batch inference over clinical text samples.")
    p.add_argument("-i", "--input",    type=Path, default=Path("data"),    metavar="DIR",
                   help="Root input directory containing {lang}/ subdirs (default: data/)")
    p.add_argument("-o", "--output",   type=Path, default=Path("results"), metavar="DIR",
                   help="Root output directory for raw/ and formatted/ results (default: results/)")
    p.add_argument("-l", "--langs",    nargs="+", default=None,            metavar="LANG",
                   help="Language codes to process, e.g. en es cz (default: all found in input dir)")
    p.add_argument("-e", "--entities", nargs="+", default=None,            metavar="ENTITY",
                   help="Entity types to run, e.g. disease symptom (default: all registered per language)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.input.exists():
        log.error("Input directory %s not found.", args.input)
        sys.exit(1)

    from app.model_manager.resolver import LocalResolver
    from app.src.format import Dt4hFormatter
    from app.src.ner import encoder_inference
    from app.utils.results_postprocessing import join_all_entities

    try:
        import torch
        _has_cuda = torch.cuda.is_available()
    except ImportError:
        _has_cuda = False

    log.info("Input : %s", args.input.resolve())
    log.info("Output: %s", args.output.resolve())
    if args.entities:
        log.info("Entity filter: %s", ", ".join(args.entities))

    resolver = LocalResolver()
    formatter = Dt4hFormatter()

    if args.langs:
        lang_dirs = sorted(args.input / lang for lang in args.langs if (args.input / lang).is_dir())
        missing = [l for l in args.langs if not (args.input / l).is_dir()]
        for l in missing:
            log.warning("Language dir %s/%s not found — skip.", args.input, l)
    else:
        lang_dirs = sorted(p for p in args.input.iterdir() if p.is_dir())

    if not lang_dirs:
        log.error("No language directories found under %s.", args.input)
        sys.exit(1)

    log.info("Languages to process: %s", ", ".join(d.name for d in lang_dirs))

    for lang_dir in lang_dirs:
        lang = lang_dir.name
        log.info("")
        log.info("━━━  %s  ━━━", lang.upper())

        model_paths = _check_registry(resolver, lang, entity_filter=args.entities)
        if model_paths is None:
            continue
        log.info("[%s] Models loaded: %d", lang, len(model_paths))

        txt_files = sorted(lang_dir.glob("*.txt"))
        if not txt_files:
            log.warning("[%s] No .txt files found — skip.", lang)
            continue
        log.info("[%s] Input files : %d", lang, len(txt_files))

        texts = [f.read_text(encoding="utf-8") for f in txt_files]
        log.info("[%s] Running NER inference...", lang)

        raw = encoder_inference(texts=texts, ner_models=model_paths, version=2)
        flat = join_all_entities(raw)  # [n_texts][n_entities]

        total_ann = sum(len(a) for a in flat)
        log.info("[%s] Inference done — %d annotation(s) found", lang, total_ann)

        ann_rows: list[tuple[str, dict]] = []
        for txt_file, text, annotations in zip(txt_files, texts, flat):
            stem = txt_file.stem
            _write_tsv(args.output / lang / "raw" / f"{stem}.ann", annotations)
            _write_json(args.output / lang / "formatted" / f"{stem}.json", text, annotations, formatter)
            log.info("[%s]   %s → %d annotation(s)", lang, txt_file.name, len(annotations))
            for ann in annotations:
                ann_rows.append((txt_file.name, ann))

        tsv_path = args.output / lang / f"{lang}.tsv"
        _write_ann(tsv_path, ann_rows)
        log.info("[%s] Combined TSV → %s", lang, tsv_path)
        log.info("[%s] Done.", lang)

        del raw, flat, texts
        gc.collect()
        if _has_cuda:
            torch.cuda.empty_cache()

    log.info("")
    log.info("All languages processed.")


if __name__ == "__main__":
    main()
