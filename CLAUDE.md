# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install uv
uv sync

# Run server
uv run flask run --host=0.0.0.0 --port=5000

# Pre-flight validation (also pre-builds missing vector DBs)
uv run test_init.py

# HTTP-level endpoint tests (server must be running)
uv run test_api.py
uv run test_api.py --validation-only   # no models needed, fast
uv run test_api.py --url http://hostname:5000

# Docker
docker compose up
docker compose build --no-cache && docker compose up
```

## Architecture

Flask API that chains **NER → NEL → (optional) Negation** into a single pipeline.

### Request flow

```
POST /annotate or /annotate_dir
  → input validation (_extract_pipeline_params)
  → _build_pipeline() — keyed by (method, lang, frozenset(entities), negation)
     cached in module-level _pipeline_cache; built once per unique combination
  → pipeline.predict(texts)
     ├─ NER: HuggingFace token-classification (encoder_inference, v2)
     │       sentence splitting + max-length chunking
     ├─ NEL: selected by method
     │       biencoder: SentenceTransformer → cosine over pre-built .pt vector DB
     │       bm25/fuzzy: lexical matching over gazetteer TSV
     │       lookup: direct dict lookup, no NER step
     └─ Negation (optional, biencoder only):
             dedicated NER model → overlap detection adds is_negated/is_uncertain
  → join_all_entities() — merge contiguous same-class spans, sort by offset
  → PassthroughFormatter.serialize() → {metadata, annotations, processing_success, processing_date}
  → JSON response or write to output_dir
```

### Key files

| Path | Role |
|---|---|
| `app/__init__.py` | Flask app, all endpoints, pipeline cache, shared helpers |
| `app/config.py` | `REGISTRY_PATH` and `RESOURCES_PATH` — change these to point at your registry |
| `app/src/pipelines.py` | `LookupPipeline`, `FuzzyMatchPipeline`, `BM25OkapiPipeline`, `BiencoderPipeline` |
| `app/model_manager/resolver.py` | `LocalResolver` — single source of truth for all resource paths |
| `app/model_manager/default_registry.yaml` | Template registry with HuggingFace repo IDs for all supported languages |
| `app/utils/results_postprocessing.py` | `merge_contiguous_entities`, `join_all_entities` |

### Registry and resource layout

`app/config.py` points to the active registry YAML (`REGISTRY_PATH`) and resource root (`RESOURCES_PATH`). The current config uses `toy_registry.yaml` — copy and rename for production use.

`LocalResolver` derives all on-disk paths from the registry:
- NER models: `{RESOURCES_PATH}/local_models/ner_models/{lang}/{entity}/{model_name}/`
- NEL models: `{RESOURCES_PATH}/local_models/nel_models/{lang}/{model_name}/`
- Gazetteers: absolute paths specified directly in the registry (must be TSV with `term` + `code` columns)
- Vector DBs: `{RESOURCES_PATH}/vectorized_dbs/{lang}/{entity}_{nel_model_name}.pt` — auto-built on first request; swapping the NEL model triggers a rebuild

When `local_path` is `null` in the registry, the resolver returns the target download path and a `repo_id`; the downloader then fetches from HuggingFace. Once downloaded, `local_path` is written back.

### Adding a new language or entity type

1. Add NER entry under `ner.<lang>.<entity>` in the registry YAML with `repo_id` and `local_path: null`.
2. Add NEL entry under `nel.<lang>` if not present.
3. Add gazetteer absolute path under `gazetteers.<lang>.<entity>`.
4. Add `vectorized_dbs.<lang>.<entity>: null` — the DB is built on first `biencoder` request.
5. Run `uv run test_init.py` to pre-build the vector DB before serving.

### Negation constraint

`negation: true` is only valid with `method: "biencoder"`. The negation NER model must be registered under `ner.<lang>.negation`. Currently only `es` has a negation model in the default registry.
