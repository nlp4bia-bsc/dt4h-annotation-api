"""
run_nerl.py — batch NER (+ optional NEL) inference over data/{lang}/ samples.

Two modes, selected by ``--nel``:

  * **NER only** (default) — token classification, nothing else.  Annotations
    carry no ``code``/``term``/``nel_score``; the CDM output emits nulls there.
  * **NER + NEL** (``--nel``) — the same ``BiencoderPipeline`` the Flask
    ``/process_bulk`` endpoint uses, minus negation: NER, then dense retrieval
    against the per-entity FAISS index, so every annotation carries a
    normalised ``code``, its canonical ``term`` and a ``nel_score``.

``--nel`` needs more on disk than the default mode: the NEL encoder, a
gazetteer per entity type, and a built vector DB per entity type.  Missing
resources abort *that language* with a message naming each one; the run
continues with the next.  Pre-build the indexes with ``uv run test_init.py``.

Usage:
    uv run run_nerl.py
    uv run run_nerl.py --nel -l es -e disease symptom

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
  3. Runs inference across all registered entity types (disease, symptom, etc.) —
     NER alone, or NER + NEL under ``--nel``.
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

def _check_registry(
    resolver, lang: str, entity_filter: list[str] | None = None
) -> tuple[list[str], list[Path]] | None:
    """
    Return ``(entity_types, ner_model_paths)`` for *lang*, or None if any are missing.

    Reads local_path from the registry YAML directly: a null entry means the
    model has not been downloaded yet, so the whole language is skipped.
    If *entity_filter* is given, only those entity types are checked/used.

    The entity names come back alongside the paths because ``--nel`` builds a
    ``BiencoderPipeline``, which resolves its own model paths and needs the
    names; both lists are in the same order.
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

    return entity_types, model_paths


def _check_nel_registry(resolver, lang: str, entity_types: list[str]) -> bool:
    """
    Return True when every NEL resource for *lang* is present.

    ``BiencoderPipeline`` runs equivalent checks and raises one RuntimeError
    listing what it could not find, but that arrives as a single opaque blob
    partway through a run. Checking here instead keeps a NEL failure in the same
    per-resource, ``[lang] ... — skip.`` form the NER check above uses, and names
    the command that fixes each one.

    Unlike ``_check_registry``, this reports *every* problem before giving up
    rather than returning on the first: a missing vector DB and a missing
    gazetteer are fixed by different commands, and finding that out one run at a
    time is the slow way to learn it.
    """
    from app.model_manager.resolver import ModelNotFoundError

    problems: list[str] = []

    # One NEL encoder is shared across every entity type for the language.
    try:
        _, repo_id = resolver.get_nel_path(lang)
        if repo_id is not None:
            problems.append(
                "NEL model not downloaded — run 'uv run python -m app.model_manager'."
            )
    except (ModelNotFoundError, FileNotFoundError) as exc:
        problems.append(f"NEL model unavailable: {exc}")

    # Gazetteer and vector DB are per entity type, and the index is keyed by the
    # NEL model name — swapping that model reports every index as unbuilt.
    for entity in entity_types:
        try:
            resolver.get_gaz_path(lang, entity)
        except (ModelNotFoundError, FileNotFoundError) as exc:
            problems.append(f"'{entity}' gazetteer unavailable: {exc}")

        try:
            _, built = resolver.get_vector_db_path(lang, entity)
            if not built:
                problems.append(
                    f"'{entity}' vector DB not built — run 'uv run test_init.py' to build it."
                )
        except (ModelNotFoundError, FileNotFoundError) as exc:
            problems.append(f"'{entity}' vector DB unavailable: {exc}")

    for problem in problems:
        log.warning("[%s] %s", lang, problem)
    if problems:
        log.warning("[%s] %d NEL resource(s) unavailable — skip.", lang, len(problems))
        return False

    return True


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


def _write_tsv(path: Path, rows: list[tuple[str, dict]], with_nel: bool = False) -> None:
    """Write the per-language annotation table.

    Under ``--nel`` three linking columns are appended.  They are not merely
    left blank in NER-only mode: a column that is always empty invites the
    reader to treat a missing code as an unlinked mention rather than as a
    stage that never ran.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: (r[0], r[1]["start"]))
    header = ["filename", "label", "start", "end", "span", "ner_score"]
    if with_nel:
        header += ["code", "term", "nel_score"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(header)
        for filename, ann in sorted_rows:
            row = [filename, ann["ner_class"], ann["start"], ann["end"], ann["span"], ann["ner_score"]]
            if with_nel:
                row += [ann.get("code"), ann.get("term"), ann.get("nel_score")]
            writer.writerow(row)


def _write_ann(path: Path, annotations: list[dict], with_nel: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")

        for ann in annotations:
            # Just pass the values as a list in the order you want them
            row = [
                ann["ner_class"],
                ann["start"],
                ann["end"],
                ann["span"],
            ]
            if with_nel:
                row += [ann.get("code"), ann.get("term")]
            writer.writerow(row)


def _write_json(path: Path, text: str, annotations: list[dict], footer: dict, formatter) -> None:
    """Serialise one document to CDM JSON.

    NER-only annotations carry no linking fields; ``Dt4hFormatter`` treats those
    as optional and emits nulls, so no stubbing is needed here.  Under ``--nel``
    the same call fills ``controlled_vocabulary_concept_identifier``,
    ``..._official_term`` and ``concept_confidence`` from the linker output.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    result = formatter.serialize(text, annotations, footer)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch NER (optionally + NEL) inference over clinical text samples.")
    p.add_argument("-i", "--input",    type=Path, default=Path("data"),    metavar="DIR",
                   help="Root input directory containing {lang}/ subdirs (default: data/)")
    p.add_argument("-o", "--output",   type=Path, default=Path("results"), metavar="DIR",
                   help="Root output directory for raw/ and formatted/ results (default: results/)")
    p.add_argument("-l", "--langs",    nargs="+", default=None,            metavar="LANG",
                   help="Language codes to process, e.g. en es cz (default: all found in input dir)")
    p.add_argument("-e", "--entities", nargs="+", default=None,            metavar="ENTITY",
                   help="Entity types to run, e.g. disease symptom (default: all registered per language)")
    p.add_argument("--nel",            action="store_true",
                   help="Link each entity to a gazetteer code after NER (adds code/term/nel_score). "
                        "Requires the NEL model, a gazetteer and a built vector DB per entity type; "
                        "pre-build with 'uv run test_init.py'. Default: NER only.")
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

    # Deferred: importing the pipeline module pulls in the NEL and negation
    # stacks, which an NER-only run has no use for.
    if args.nel:
        from app.src.pipelines import BiencoderPipeline

    try:
        import torch
        _has_cuda = torch.cuda.is_available()
    except ImportError:
        _has_cuda = False

    log.info("Input : %s", args.input.resolve())
    log.info("Output: %s", args.output.resolve())
    log.info("Stages: %s", "NER + NEL" if args.nel else "NER only")
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

        registered = _check_registry(resolver, lang, entity_filter=args.entities)
        if registered is None:
            continue
        entity_types, model_paths = registered
        log.info("[%s] Models loaded: %d (%s)", lang, len(model_paths), ", ".join(entity_types))

        # Checked here, beside the NER check and before any file is read, so a
        # language that cannot be linked says so up front instead of after
        # loading and validating every document in it.
        if args.nel and not _check_nel_registry(resolver, lang, entity_types):
            continue

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

        pipeline = None
        if args.nel:
            # Built here, not before the documents are loaded: opening the FAISS
            # index and the NEL encoder is the expensive part, and a language
            # with nothing to annotate should not pay for it.
            log.info("[%s] Loading NEL resources (%d gazetteer/index pair(s))...",
                     lang, len(entity_types))
            try:
                pipeline = BiencoderPipeline(
                    lang=lang, entities=entity_types, negation=False, ner_version=2
                )
            except Exception as exc:
                # Everything _check_nel_registry can see was already checked, so
                # a failure here is the index itself: most often a manifest whose
                # gazetteer SHA no longer matches, i.e. the TSV was edited after
                # the index was built. The exception type is logged because that
                # is the part distinguishing a stale index from an unloadable
                # model, and the run continues with the next language.
                log.error("[%s] NEL pipeline failed to load — %s: %s", lang, type(exc).__name__, exc)
                log.error("[%s] If a gazetteer changed, rebuild its index with 'uv run test_init.py' — skip.", lang)
                log.debug("[%s] NEL pipeline traceback:", lang, exc_info=True)
                continue

        log.info("[%s] Running %s inference on %d document(s)...",
                 lang, "NER + NEL" if args.nel else "NER", len(texts))

        raw = None
        if pipeline is not None:
            flat = pipeline.predict(texts)  # [n_texts][n_entities], already joined
        else:
            raw = encoder_inference(texts=texts, ner_models=model_paths, version=2)
            flat = join_all_entities(raw)  # [n_texts][n_entities]

        total_ann = sum(len(a) for a in flat)
        log.info("[%s] Inference done — %d annotation(s) found", lang, total_ann)

        ann_rows: list[tuple[str, dict]] = []
        for (json_file, text, metadata), annotations in zip(loaded, flat):
            stem = json_file.stem
            _write_ann(args.output / lang / "raw" / f"{stem}.ann", annotations, with_nel=args.nel)
            _write_json(args.output / lang / "formatted" / f"{stem}.json",
                        text, annotations, metadata, formatter)
            log.info("[%s]   %s → %d annotation(s)", lang, json_file.name, len(annotations))
            for ann in annotations:
                ann_rows.append((stem, ann))
        processed_total += len(loaded)

        tsv_path = args.output / lang / f"{lang}.tsv"
        _write_tsv(tsv_path, ann_rows, with_nel=args.nel)
        log.info("[%s] Combined TSV → %s", lang, tsv_path)
        log.info("[%s] Done.", lang)

        # The pipeline holds the FAISS index and the NEL encoder; dropping it
        # is what makes the next language start from a clean allocation.
        del raw, flat, texts, loaded, pipeline
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
