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

# Batch NER over CDM JSON documents in data/{lang}/
uv run run_ner.py
uv run run_ner.py -i data -o results -l es en -e disease symptom

# Run server
uv run flask run --host=0.0.0.0 --port=5000

# Docker
docker compose up
docker compose build --no-cache && docker compose up
```

## Two entry points

The repository serves the same pipeline through two independent front ends.
They share the pipeline and formatter code but nothing else.

| | `app/__init__.py` (Flask) | `run_ner.py` (CLI) |
|---|---|---|
| Transport | HTTP, port 5000 | local filesystem |
| Input | CogStack envelope in the request body | CDM JSON files in `data/{lang}/` |
| Stages | NER → NEL → optional negation | NER only |
| Intended caller | CogStack/NiFi on `cogstack-net` | run by hand |

`run_ner.py` does not call the API. It imports the pipeline directly.

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
     ├─ NEL: biencoder — SentenceTransformer → cosine over pre-built .pt vector DB
     └─ Negation (optional): dedicated NER model → overlap detection adds
             is_negated / is_uncertain
  → join_all_entities() — merge contiguous same-class spans, sort by offset
  → Dt4hFormatter.serialize() → CDM NlpResponse
  → JSON array, one NlpResponse per input item
```

Endpoints, all of them: `GET /` (health), `POST /process_bulk`, `POST /sync_models`.

## run_ner.py input contract

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

## Key files

| Path | Role |
|---|---|
| `app/__init__.py` | Flask app, all endpoints, pipeline cache |
| `app/config.py` | `REGISTRY_PATH`, `RESOURCES_PATH`, device selection |
| `app/src/pipelines.py` | `LookupPipeline`, `FuzzyMatchPipeline`, `BM25OkapiPipeline`, `BiencoderPipeline` |
| `app/src/format/data_structures.py` | Pydantic models for the CDM — input and output |
| `app/src/format/dt4h.py` | `Dt4hFormatter` — raw pipeline output → CDM |
| `app/model_manager/resolver.py` | `LocalResolver` — single source of truth for all resource paths |
| `app/model_manager/default_registry.yaml` | Template registry with HuggingFace repo IDs |
| `app/utils/results_postprocessing.py` | `merge_contiguous_entities`, `join_all_entities` |
| `run_ner.py` | Batch NER CLI over CDM JSON |
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

`concept_confidence` carries the **NER** score. The CDM has one confidence slot
per annotation and no home for the NEL score — see `docs/cdm_open_questions.md`.

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
- Vector DBs: `{RESOURCES_PATH}/vectorized_dbs/{lang}/{entity}_{nel_model_name}.pt`
  — auto-built on first request; swapping the NEL model triggers a rebuild

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

## Negation constraint

`negation: true` is only valid with `method: "biencoder"`. The negation NER model
must be registered under `ner.<lang>.negation`. In `default_registry.yaml` every
language has a `negation` entry, but most have `repo_id: null`.
