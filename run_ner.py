"""
run_ner.py — NER-only batch inference over data/{lang}/ samples.

Usage:
    uv run run_ner.py

Input format
------------
Input documents are DT4H CDM JSON files — the same schema this script emits, with
``annotations`` left to be filled in.  Only ``nlp_output.record_metadata`` is
read; any ``annotations``, ``processing_success`` or ``nlp_service_info`` keys
present in the input are ignored.

The document text may be supplied in either of two ways:

  * inline, as ``nlp_output.record_metadata.text``; or
  * in a sibling plain-text file — ``doc1.json`` pairs with ``doc1.txt`` in the
    same directory — when the inline ``text`` is empty or absent.

A document is **rejected** (logged, skipped, counted; the run continues) when:

  * the JSON is malformed;
  * any CDM-mandatory metadata field is missing — clinical_site_id, patient_id,
    admission_id, record_id, deidentified, deidentification_pipeline_name,
    deidentification_pipeline_version;
  * ``report_language`` disagrees with the language directory the file sits in;
  * the text is empty inline and no readable sibling ``.txt`` exists.

For each language directory found in data/:
  1. Checks the registry: all NER models must have a local_path (i.e. downloaded).
  2. Loads all .json files from data/{lang}/, resolving text as described above.
  3. Runs NER inference across all registered entity types (disease, symptom, etc.).
  4. Writes results/{lang}/raw/{stem}.ann   — flat entity spans per document.
  5. Writes results/{lang}/formatted/{stem}.json — DT4H CDM JSON per document.
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


def _load_document(json_path: Path, lang: str) -> tuple[str, dict] | None:
    """Load one CDM input document, resolving its text.

    Returns ``(text, metadata)`` where *metadata* is the record metadata as a
    plain dict suitable for use as a formatter footer, or ``None`` if the
    document was rejected.  Every rejection path logs the reason; callers count
    the ``None`` results and carry on with the rest of the batch.

    Text resolution: the inline ``record_metadata.text`` is used when non-blank,
    otherwise the sibling ``{stem}.txt`` is read from the same directory.
    """
    from pydantic import ValidationError

    from app.src.format.data_structures import NlpInputDocument

    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.error("[%s] %s: malformed JSON (%s) — skip.", lang, json_path.name, exc)
        return None
    except OSError as exc:
        log.error("[%s] %s: unreadable (%s) — skip.", lang, json_path.name, exc)
        return None

    if not isinstance(raw, dict):
        log.error(
            "[%s] %s: top-level value is %s, expected a CDM object — skip.",
            lang, json_path.name, type(raw).__name__,
        )
        return None

    try:
        document = NlpInputDocument(**raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        log.error("[%s] %s: invalid CDM metadata — %s — skip.", lang, json_path.name, problems)
        return None

    metadata = document.nlp_output.record_metadata

    # The language directory is authoritative — it selects the models. A record
    # claiming a different language is a packaging error, not something to guess at.
    if metadata.report_language is not None and metadata.report_language != lang:
        log.error(
            "[%s] %s: report_language is '%s' but the file is in the '%s' directory — skip.",
            lang, json_path.name, metadata.report_language, lang,
        )
        return None

    text = metadata.text
    if not text.strip():
        sidecar = json_path.with_suffix(".txt")
        if not sidecar.is_file():
            log.error(
                "[%s] %s: 'text' is empty and no sibling %s exists — skip.",
                lang, json_path.name, sidecar.name,
            )
            return None
        try:
            text = sidecar.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            log.error("[%s] %s: cannot read %s (%s) — skip.", lang, json_path.name, sidecar.name, exc)
            return None
        if not text.strip():
            log.error("[%s] %s: sibling %s is empty — skip.", lang, json_path.name, sidecar.name)
            return None
        log.debug("[%s] %s: text read from %s", lang, json_path.name, sidecar.name)

    return text, metadata.model_dump()


def _write_tsv(path: Path, rows: list[tuple[str, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: (r[0], r[1]["start"]))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["filename", "label", "start", "end", "span", "ner_score"])
        for filename, ann in sorted_rows:
            writer.writerow([filename, ann["ner_class"], ann["start"], ann["end"], ann["span"], ann["ner_score"]])


def _write_ann(path: Path, annotations: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        
        for ann in annotations:
            # Just pass the values as a list in the order you want them
            writer.writerow([
                ann["ner_class"],
                ann["start"],
                ann["end"],
                ann["span"],
            ])


def _write_json(path: Path, text: str, annotations: list[dict], footer: dict, formatter) -> None:
    """Serialise one document to CDM JSON.

    NER-only annotations carry no linking fields; ``Dt4hFormatter`` treats those
    as optional and emits nulls, so no stubbing is needed here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    result = formatter.serialize(text, annotations, footer)
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

    processed_total = 0
    rejected_total = 0

    for lang_dir in lang_dirs:
        lang = lang_dir.name
        log.info("")
        log.info("━━━  %s  ━━━", lang.upper())

        model_paths = _check_registry(resolver, lang, entity_filter=args.entities)
        if model_paths is None:
            continue
        log.info("[%s] Models loaded: %d", lang, len(model_paths))

        json_files = sorted(lang_dir.glob("*.json"))
        if not json_files:
            log.warning("[%s] No .json files found — skip.", lang)
            continue
        log.info("[%s] Input files : %d", lang, len(json_files))

        loaded: list[tuple[Path, str, dict]] = []
        for json_file in json_files:
            document = _load_document(json_file, lang)
            if document is None:
                rejected_total += 1
                continue
            text, metadata = document
            loaded.append((json_file, text, metadata))

        n_rejected = len(json_files) - len(loaded)
        if n_rejected:
            log.warning("[%s] Rejected %d of %d document(s) — see errors above.",
                        lang, n_rejected, len(json_files))
        if not loaded:
            log.warning("[%s] No usable documents — skip.", lang)
            continue

        texts = [text for _, text, _ in loaded]
        log.info("[%s] Running NER inference on %d document(s)...", lang, len(texts))

        raw = encoder_inference(texts=texts, ner_models=model_paths, version=2)
        flat = join_all_entities(raw)  # [n_texts][n_entities]

        total_ann = sum(len(a) for a in flat)
        log.info("[%s] Inference done — %d annotation(s) found", lang, total_ann)

        ann_rows: list[tuple[str, dict]] = []
        for (json_file, text, metadata), annotations in zip(loaded, flat):
            stem = json_file.stem
            _write_ann(args.output / lang / "raw" / f"{stem}.ann", annotations)
            _write_json(args.output / lang / "formatted" / f"{stem}.json",
                        text, annotations, metadata, formatter)
            log.info("[%s]   %s → %d annotation(s)", lang, json_file.name, len(annotations))
            for ann in annotations:
                ann_rows.append((stem, ann))
        processed_total += len(loaded)

        tsv_path = args.output / lang / f"{lang}.tsv"
        _write_tsv(tsv_path, ann_rows)
        log.info("[%s] Combined TSV → %s", lang, tsv_path)
        log.info("[%s] Done.", lang)

        del raw, flat, texts, loaded
        gc.collect()
        if _has_cuda:
            torch.cuda.empty_cache()

    log.info("")
    if rejected_total:
        log.warning("All languages processed — %d document(s) annotated, %d rejected.",
                    processed_total, rejected_total)
    else:
        log.info("All languages processed — %d document(s) annotated.", processed_total)


if __name__ == "__main__":
    main()
