# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install uv
uv sync

# Create the active registry (config.py points at registry.yaml, which is gitignored)
cp app/model_manager/default_registry.yaml app/model_manager/registry.yaml

# Download all models listed in the registry
uv run python -m app.model_manager

# Pre-flight validation (also pre-builds missing vector DBs)
uv run test_init.py

# Test suite — offline, ~1s, needs no registry.yaml, resources, or model download
uv run pytest

# Batch NER over CDM JSON documents in data/{lang}/
uv run run_nerl.py
uv run run_nerl.py -i data -o results -l es en -e disease symptom

# Same, with entity linking (needs gazetteers + built vector DBs)
uv run run_nerl.py --nel -l es -e disease symptom

# Run server
uv run flask run --host=0.0.0.0 --port=5000

# Docker
docker compose up
docker compose build --no-cache && docker compose up
```

## Two entry points

The repository serves the same pipeline through two independent front ends.
They share the pipeline and formatter code but nothing else.

| | `app/__init__.py` (Flask) | `run_nerl.py` (CLI) |
|---|---|---|
| Transport | HTTP, port 5000 | local filesystem |
| Input | CogStack envelope in the request body | CDM JSON files in `data/{lang}/` |
| Stages | NER → NEL → optional negation | NER, plus NEL under `--nel` |
| Intended caller | CogStack/NiFi on `cogstack-net` | run by hand |

`run_nerl.py` does not call the API. It imports the pipeline directly.
Under `--nel` it builds the same `BiencoderPipeline` the Flask route uses, with
`negation=False`; the default mode calls `encoder_inference` alone and never
imports the NEL stack.

## Flask request flow

```
POST /process_bulk?language=es&entities=disease,symptom&negation=false
  → query params parsed (language, entities, negation — all from the URL)
  → body parsed: {"content": [{"id", "text", "footer"}, ...]}
     'text' is required and must be non-empty; 'id' is currently unused
  → pipeline cache, keyed by (method, lang, frozenset(entities), negation)
     module-level _pipeline_cache; method is hardcoded 'biencoder'
  → pipeline.predict(texts)
     ├─ NER: HuggingFace token-classification (encoder_inference, v2)
     │       sentence splitting + max-length chunking
     ├─ NEL: EntityLinker, one per entity type, built in the pipeline's
     │       __init__ and reused across requests
     │       ├─ DenseGenerator: SentenceTransformer → FAISS IndexFlatIP
     │       │   (exact cosine) over a persisted vector DB
     │       ├─ ExactMatchGenerator: normalised surface-form lookup
     │       ├─ TfidfCharNgramGenerator / BM25Generator: shared persisted
     │       │   sparse index, built lazily on first use
     │       │   (all three are OFF by default — see BiencoderPipeline flags)
     │       ├─ RRF fusion, only when more than one generator is active.
     │       │   nel_score is never the RRF score: it reports the similarity
     │       │   from whichever generator ranked the winning code best,
     │       │   clamped to [0, 1] (EntityLinker._reportable_score)
     │       └─ CrossEncoderReranker over the shortlist — OFF by default and
     │           unconfigured; when it runs it owns both the final order and
     │           the reported score
     └─ Negation (optional): dedicated NER model → overlap detection adds
             is_negated / is_uncertain
  → join_all_entities() — merge contiguous same-class spans, sort by offset
  → Dt4hFormatter.serialize() → CDM NlpResponse
  → JSON array, one NlpResponse per input item
```

Endpoints, all of them: `GET /` (health), `POST /process_bulk`, `POST /sync_models`.

## run_nerl.py input contract

Input files are CDM JSON — the same schema the script emits, with `annotations`
left to be filled in. Only `nlp_output.record_metadata` is read; `annotations`,
`processing_success` and `nlp_service_info` in the input are ignored.

Text comes from one of two places:

- inline, as `nlp_output.record_metadata.text`; or
- from a sibling `.txt` file (`doc1.json` pairs with `doc1.txt` in the same
  directory) when the inline `text` is empty, blank, or absent.

A document is rejected — logged, skipped, counted, run continues — when the JSON
is malformed, a CDM-mandatory field is missing, `report_language` disagrees with
the language directory, or the text is empty with no readable sidecar.

The seven mandatory metadata fields are `clinical_site_id`, `patient_id`,
`admission_id`, `record_id`, `deidentified`, `deidentification_pipeline_name`,
`deidentification_pipeline_version` (see `MANDATORY_INPUT_FIELDS`).

The language directory name is authoritative — it selects the models.

Outputs per language: `results/{lang}/raw/{stem}.ann`,
`results/{lang}/formatted/{stem}.json`, `results/{lang}/{lang}.tsv`.

`--nel` appends `code` / `term` to the `.ann` rows and `code` / `term` /
`nel_score` to the TSV; the columns are absent without it rather than empty, so
a blank code never has to be read as "unlinked" when the stage never ran. The
CDM JSON needs no switch — `Dt4hFormatter` fills the linking fields when they
are present and emits nulls when they are not.

Missing NEL resources abort that one language (message names each missing item)
and the run continues; a language whose documents were all rejected never loads
the index at all.

## Key files

| Path | Role |
|---|---|
| `app/__init__.py` | Flask app, all endpoints, pipeline cache |
| `app/config.py` | `REGISTRY_PATH`, `RESOURCES_PATH`, device selection |
| `app/src/pipelines.py` | `LookupPipeline`, `FuzzyMatchPipeline`, `BM25OkapiPipeline`, `BiencoderPipeline` |
| `app/src/nel/linker.py` | `EntityLinker` — owns the generators, fuses, picks the best candidate |
| `app/src/nel/vector_store.py` | Persisted FAISS index: build, manifest validation, cached load |
| `app/src/nel/gazetteer.py` | `load_gazetteer` — the only way either side derives index row order |
| `app/src/nel/candidates.py` | `DenseGenerator`, `ExactMatchGenerator`, `TfidfCharNgramGenerator`, `BM25Generator` and the generator Protocol |
| `app/src/nel/lexical_index.py` | Persisted sparse index shared by the TF-IDF and BM25 generators |
| `app/src/nel/fusion.py` | `reciprocal_rank_fusion` |
| `app/src/nel/rerank.py` | `CrossEncoderReranker` — inference only, no training |
| `tests/conftest.py` | `StubEncoder` and the gazetteer / FAISS-index fixtures |
| `app/src/format/data_structures.py` | Pydantic models for the CDM — input and output |
| `app/src/format/dt4h.py` | `Dt4hFormatter` — raw pipeline output → CDM |
| `app/model_manager/resolver.py` | `LocalResolver` — single source of truth for all resource paths |
| `app/model_manager/default_registry.yaml` | Template registry with HuggingFace repo IDs |
| `app/utils/results_postprocessing.py` | `merge_contiguous_entities`, `join_all_entities` |
| `run_nerl.py` | Batch CDM JSON CLI — NER, or NER + NEL under `--nel` |
| `docs/cdm_open_questions.md` | Deferred CDM decisions and known gaps |

## CDM models

`data_structures.py` holds two families:

- **Output** — `RecordMetadata`, `Annotation`, `NlpOutput`, `NlpServiceInfo`,
  `NlpResponse`. Permissive: nearly every field optional, so NER-only runs and
  sparse footers still serialise.
- **Input** — `RecordMetadataInput`, `NlpInputDocument`. Strict: the seven
  mandatory fields are required, `text` may be empty (meaning "read the
  sidecar"), and the `nlp_processing_*` fields are absent.

Controlled-vocabulary fields are typed `str` with an `AfterValidator` that logs a
warning on out-of-vocabulary values rather than raising. Unexpected values from
models or source records are surfaced, not fatal. Use `_controlled()` /
`_controlled_required()` when adding such a field.

`concept_confidence` carries the **NEL** score. The CDM has one confidence slot
per annotation and no home for the NER score — see `docs/cdm_open_questions.md`.
NER-only runs (`run_nerl.py` without `--nel`) emit `concept_confidence: null`.

`negation` is `null` when the negation model was not run, and only `"yes"`/`"no"`
when an entity was actually assessed. Do not default it to `"no"`.

## Registry and resource layout

`app/config.py` points at the active registry YAML (`REGISTRY_PATH`) and resource
root (`RESOURCES_PATH`). Only `default_registry.yaml` is committed; copy it to
`registry.yaml` before first run. `LocalResolver._import_registry` returns `{}`
on a missing file with only a log line, so an absent registry looks like "no
languages registered" rather than an error.

`LocalResolver` derives all on-disk paths:

- NER models: `{RESOURCES_PATH}/local_models/ner_models/{entity}/{model_name}/`
- NEL models: `{RESOURCES_PATH}/local_models/nel_models/{model_name}/`
- Gazetteers: absolute paths given directly in the registry (TSV with `term` + `code`)
- Vector DBs: `{RESOURCES_PATH}/vectorized_dbs/{lang}/{entity}_{nel_model_name}.faiss`
  — auto-built on first request; swapping the NEL model triggers a rebuild.
  Each index has a sibling `.faiss.manifest.json` pinning the gazetteer SHA-256,
  model name, embedding dim and row count. Editing a gazetteer invalidates its
  index: loading raises rather than returning codes from shifted rows.
  Legacy `.pt` memmaps convert without re-encoding via
  `uv run python scripts/migrate_vector_db_to_faiss.py`.
- Lexical indexes: `{RESOURCES_PATH}/vectorized_dbs/{lang}/{entity}.lexical.pkl`
  — not registry-backed, built lazily on first use, keyed by gazetteer only
  (no NEL model). Its manifest pins the gazetteer SHA-256 plus the
  scikit-learn and scipy versions; any mismatch rebuilds rather than
  unpickling a stale estimator.

Models sharing a `repo_id` across languages share one on-disk copy.
`ModelManager.sanitize()` downloads each unique `(repo_id, path)` pair once and
writes the shared path back to every matching registry entry.

When `local_path` is `null`, the resolver returns the target download path plus a
`repo_id`, and the downloader fetches from HuggingFace, writing `local_path` back.

Registered entity types are `disease`, `symptom`, `procedure`, `drug`, plus
`negation`. Note `drug` here vs `medication` in the CDM's `concept_class`.

## Adding a new language or entity type

1. Add NER entry under `ner.<lang>.<entity>` with `repo_id` and `local_path: null`.
2. Add NEL entry under `nel.<lang>` if not present.
3. Add gazetteer absolute path under `gazetteers.<lang>.<entity>`.
4. Add `vectorized_dbs.<lang>.<entity>: null` — built on first `biencoder` request.
5. Run `uv run test_init.py` to pre-build the vector DB before serving.

## Cross-encoder reranking

`BiencoderPipeline(..., rerank=True)` rescores the NEL shortlist with a
cross-encoder, which reads mention and candidate *together* rather than
comparing independent embeddings. Too slow to search a gazetteer with, so it
only ever sees the top-k the retrieval stage produced.

**No checkpoint ships.** `rerank.<lang>` exists for all languages in
`default_registry.yaml` with `repo_id: null`, the same convention as
`ner.<lang>.negation`. Nothing is downloaded until a `repo_id` is set, and
`rerank=True` raises `ModelNotFoundError` naming the key to fill in. Models
land in `{RESOURCES_PATH}/local_models/rerank_models/{model_name}/`.

When a reranker runs it is the authority on the final order *and* on
`nel_score`. `CrossEncoder.predict` applies the model's own activation —
sigmoid for single-logit rerankers, identity for others — so
`rerank.normalise_scores` squashes raw logits into [0, 1] monotonically, never
changing the order. The pre-rerank method, score and rank survive in
`metadata`.

Only the ~45-line inference path was ported from the reference
implementation's 777-line `rerankers/cross_encoder.py`; the rest trains models.

## Negation constraint

`negation: true` is only valid with `method: "biencoder"`. The negation NER model
must be registered under `ner.<lang>.negation`. In `default_registry.yaml` every
language has a `negation` entry, but most have `repo_id: null`.

## Tests

`uv run pytest` — 113 tests, ~1.5s, fully offline. No `registry.yaml`, no
`app/resources/`, no model download, no network.

The NEL encoder is replaced by `tests/conftest.py:StubEncoder`, which embeds
text as unit vectors seeded from a SHA-256 of the normalised string. Python's
built-in `hash` is salted per process, so seeding explicitly is what keeps
ranking assertions stable across runs. `StubEncoder(overrides=...)` pins chosen
texts to exact vectors — that is how a test scripts the dense retriever into
getting a mention *wrong*, so a lexical rescue can be asserted rather than
hoped for.

Everything else is the real code path: real FAISS indexes, real gazetteer
loading, real sklearn vectorizers, real CDM serialisation. All artifacts go to
pytest's `tmp_path`; the suite writes nothing into the repo.

An autouse fixture clears the module-level caches in `vector_store` and
`lexical_index` around every test. Without it a test can pass only because an
earlier one warmed a cache.

Keep the suite offline. A test needing a real checkpoint belongs behind an
explicit marker, not in the default run.

`tests/test_no_undefined_names.py` shells out to `ruff check --select F821`.
It exists because `FuzzyMatchPipeline`/`BM25OkapiPipeline` called an
undefined `ner_inference` for a long time without anyone noticing: neither is
reachable over HTTP, and exercising them needs real NER checkpoints the suite
will not download. Scoped to F821 deliberately — a correctness guard, not a
style gate.
