# Named Entity Recognition + Linking API

A Flask REST API that chains **Named Entity Recognition (NER)**, **Named Entity Linking (NEL)**, and optional **negation/uncertainty detection** into a single inference pipeline for clinical text. Designed for integration with CogStack/NiFi pipelines. Output follows the DT4H CDM v2 schema.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Validation](#validation)
   - [1. Download models](#1-download-models)
   - [2. Pre-flight pipeline check](#2-pre-flight-pipeline-check)
5. [Running the Server](#running-the-server)
6. [API Reference](#api-reference)
   - [GET /](#get-)
   - [POST /process_bulk](#post-process_bulk)
   - [POST /sync_models](#post-sync_models)
7. [Response Schema](#response-schema)
8. [Examples](#examples)
9. [Docker](#docker)
10. [Architecture](#architecture)

---

## Requirements

- Python 3.10+
- A CUDA-capable GPU is strongly recommended. CPU-only mode works but is significantly slower for biencoder inference and vector DB construction.

---

## Installation

```bash
pip install uv
uv sync
```

---

## Configuration

All model and resource paths are managed through a **registry YAML file**. The path to this file is set in `app/config.py`:

```python
# app/config.py
REGISTRY_PATH = "app/model_manager/registry.yaml"
RESOURCES_PATH = "app/model_manager/resources"
```

### Registry structure

The registry maps languages and entity types to local model paths and gazetteer files. Below is a complete example for Spanish (`es`) with disease and symptom entities:

```yaml
ner:
  es:
    disease:
      repo_id: BSC-NLP4BIA/bsc-bio-ehr-es-carmen-distemist
      local_path: null   # will be populated after first download
    symptoms:
      repo_id: BSC-NLP4BIA/bsc-bio-ehr-es-carmen-symptemist
      local_path: null
    negation:
      repo_id: BSC-NLP4BIA/negation-tagger
      local_path: null

nel:
  es:
    repo_id: ICB-UMA/ClinLinker-KB-GP
    local_path: null

gazetteers:
  es:
    disease: /absolute/path/to/distemist_gazetteer.tsv
    symptoms: /absolute/path/to/symptemist_gazetteer.tsv

vectorized_dbs:
  es:
    disease: null   # built automatically on first run
    symptoms: null
```

**Key points:**

- **NER models** are per entity type. Multiple languages can share the same model (e.g. a multilingual model registered under `en`, `nl`, and `sv`). When several registry entries share the same `repo_id`, the model is downloaded once and the path is reused across all of them.
- **NEL model** is shared across all entity types within a language. Again, one `repo_id` used by multiple languages is downloaded only once.
- **Disk layout** for auto-derived paths (i.e. when `local_path` is `null`):
  - NER: `{RESOURCES_PATH}/local_models/ner_models/{entity}/{model_name}/`
  - NEL: `{RESOURCES_PATH}/local_models/nel_models/{model_name}/`
  - Vector DBs: `{RESOURCES_PATH}/vectorized_dbs/{lang}/{entity}_{nel_model_name}.pt`
- **Gazetteers** must be placed manually. Each must be a TSV file with at minimum a `term` column and a `code` column. Setting a gazetteer entry to `null` is safe — the model manager and pipeline pre-flight check will skip it rather than crash. Requests for an entity with a missing or unconfigured gazetteer will fail with a clear error listing all absent resources.
- **Vector databases** are built automatically from the gazetteer + NEL model on the first request. Once built, the path is written back to the registry so subsequent startups skip the build step. To force a rebuild, set the relevant entry to `null` in the registry.
- If a model already exists locally (e.g. pre-downloaded or manually placed), set `local_path` directly and leave `repo_id: null` — no download will be attempted.
- Swapping the NEL model produces a new vector DB filename automatically, triggering a rebuild.

### Device selection

The API detects CUDA availability at startup and sets the device accordingly. No manual configuration is needed.

---

## Validation

### 1. Download models

Run the model manager to download all NER/NEL models listed in the registry (those with `repo_id` set and `local_path: null`) and validate any configured gazetteers. Entries with `null` paths (e.g. unconfigured gazetteers or missing `repo_id`) are skipped with a warning rather than aborting.

```bash
uv run python -m app.model_manager
```

### 2. Pre-flight pipeline check

After downloading models, run the standalone validation script to verify end-to-end pipeline correctness and pre-build any missing vector databases (GPU strongly preferred for this step):

```bash
uv run test_init.py
```

If any required resource is absent — NER/NEL model not downloaded, gazetteer not configured, or vector DB not built — the script fails with a single `RuntimeError` listing **all** missing items at once so you can resolve them in one pass:

```
RuntimeError: Cannot start pipeline — 3 resource(s) unavailable:
  • NER es/disease: not downloaded — run 'python -m app.model_manager'
  • NEL es: not downloaded — run 'python -m app.model_manager'
  • Vector DB es/disease: not built — run 'uv run test_init.py'
```

The same pre-flight check runs whenever a pipeline is first instantiated at inference time.

---

## Running the Server

```bash
uv run flask run --host=0.0.0.0 --port=5000
```

The health endpoint confirms the service is up:

```bash
curl http://localhost:5000/
# → OK
```

---

## API Reference

### `GET /`

Health check. Returns `200 OK` with body `OK`.

---

### `POST /process_bulk`

Annotate a batch of clinical texts. Designed for CogStack/NiFi: inference parameters come from URL query params (flowfile attributes) and the payload follows the CogStack envelope format.

#### Query parameters

| Param | Required | Description |
|---|---|---|
| `language` | yes | Language code (e.g. `"es"`). |
| `entities` | yes | Comma-separated entity types (e.g. `"disease,symptoms"`). Must match registry entries. |
| `negation` | no | `"true"` or `"false"` (default `"false"`). Enable negation/uncertainty detection. Requires a `negation` NER model in the registry for the given language. |

#### Request body

```json
{
  "content": [
    {
      "id":     "<document identifier>",
      "text":   "<clinical text>",
      "footer": {
        "patient_id":    "...",
        "admission_id":  "...",
        "text_path":     "...",
        "..."
      }
    }
  ]
}
```

- `text` is required per item. All other item fields (`id`, `footer`) are optional.
- `footer` fields are passed through to the CDM v2 `record_metadata` object. Fields not declared in `RecordMetadata` are silently ignored.

#### Response

Array of DT4H CDM v2 objects, one per input item (see [Response Schema](#response-schema)).

---

### `POST /sync_models`

Download pending resources from the registry and clear the pipeline cache. Call this after editing `registry.yaml` to add new models (setting `local_path: null`).

Requests block until all downloads and vector DB builds complete — this may take several minutes for large models. Concurrent calls are rejected with `409`.

#### Response

| HTTP | `status` | Meaning |
|---|---|---|
| `200` | `no_op` | Nothing pending; registry already complete |
| `200` | `done` | All pending resources downloaded; pipeline cache cleared |
| `206` | `partial_error` | Some resources failed; cache NOT cleared (existing pipelines keep serving) |
| `409` | `already_running` | Another sync is in progress |
| `500` | `error` | Unexpected exception |

Response body (all non-409 cases):

```json
{
  "status": "done",
  "started_at": "2026-05-15T10:00:00+00:00",
  "finished_at": "2026-05-15T10:07:32+00:00",
  "pending_before": [
    {"resource": "ner", "lang": "es", "task": "disease", "repo_id": "BSC-NLP4BIA/..."}
  ],
  "completed": [ ... ],
  "errors": [],
  "cache_cleared": true
}
```

#### Example

```bash
curl -X POST http://localhost:5000/sync_models
```

---

## Response Schema

Each result object follows the DT4H CDM v2 (`NlpResponse`) structure:

```json
{
  "nlp_output": {
    "record_metadata": {
      "patient_id":                       "P1",
      "admission_id":                     "A1",
      "text":                             "<original input text>",
      "nlp_processing_date":              "2026-05-08T18:13:39.693396",
      "nlp_processing_pipeline_name":     "Dt4hFormatter",
      "nlp_processing_pipeline_version":  "1.0",
      "..."
    },
    "annotations": [
      {
        "concept_class":          "symptom",
        "start_offset":           18,
        "end_offset":             24,
        "mention_string":         "fiebre",
        "extraction_confidence":  0.9999,
        "concept_str":            "fiebre",
        "concept_code":           "64882008",
        "concept_confidence":     1.0,
        "negation":               "no",
        "negation_confidence":    0.0,
        "uncertainty":            "no",
        "uncertainty_confidence": 0.0
      }
    ],
    "processing_success": true
  },
  "nlp_service_info": {
    "service_app_name":  "DT4H NLP Processor",
    "service_language":  "en",
    "service_version":   "1.0",
    "service_model":     "Dt4hFormatter"
  }
}
```

`concept_class` values: `"symptom"`, `"disorder/disease"`, `"procedure"`, `"medication"`.

---

## Examples

Start the server before running any of the examples below:

```bash
uv run flask run --host=0.0.0.0 --port=5000
```

### Health check

```bash
curl http://localhost:5000/
```

### Single text

```bash
curl -X POST "http://localhost:5000/process_bulk?language=es&entities=disease,symptoms&negation=false" \
  -H 'Content-Type: application/json' \
  -d '{
    "content": [
      {
        "id": "doc1",
        "text": "El paciente presenta fiebre alta y tos persistente.",
        "footer": {
          "patient_id":   "P1",
          "admission_id": "A1",
          "text_path":    "/data/doc1.txt"
        }
      }
    ]
  }' | python3 -m json.tool
```

### Bulk texts

```bash
curl -X POST "http://localhost:5000/process_bulk?language=es&entities=disease,symptoms&negation=false" \
  -H 'Content-Type: application/json' \
  -d '{
    "content": [
      {
        "id": "doc1",
        "text": "El paciente presenta fiebre alta y tos persistente.",
        "footer": {"patient_id": "P1", "admission_id": "A1", "text_path": "/data/doc1.txt"}
      },
      {
        "id": "doc2",
        "text": "Dolor abdominal agudo. No presenta náuseas.",
        "footer": {"patient_id": "P2", "admission_id": "A2", "text_path": "/data/doc2.txt"}
      }
    ]
  }' | python3 -m json.tool
```

### With negation detection

```bash
curl -X POST "http://localhost:5000/process_bulk?language=es&entities=disease,symptoms&negation=true" \
  -H 'Content-Type: application/json' \
  -d '{
    "content": [
      {
        "id": "doc1",
        "text": "El paciente no presenta fiebre ni tos.",
        "footer": {"patient_id": "P1", "admission_id": "A1", "text_path": "/data/doc1.txt"}
      }
    ]
  }' | python3 -m json.tool
```

Entities in a negated context will have `"negation": "yes"` and a non-zero `negation_confidence`.

---

## Docker

```bash
docker compose up
```

Or for a clean rebuild:

```bash
docker compose down
docker compose build --no-cache
docker compose up
```

The default port is `5000` and can be changed in `docker-compose.yml`.

---

## Architecture

```
POST /process_bulk?language=es&entities=disease,symptoms&negation=false
      │
      ▼
 Query param parsing (language, entities, negation)
 Body parsing ({content: [{id, text, footer}]})
      │
      ▼
 Pipeline instantiation
 (biencoder × lang × entities × negation → module-level cache → LocalResolver → model paths)
 (built once per unique parameter combination; reused on subsequent requests)
      │
      ▼
 NER  — HuggingFace token-classification model (v2 encoder)
  │        sentence splitting + max-length chunking
  │        → list of {start, end, span, ner_class, ner_score}
  │
  ▼
 NEL  — biencoder (SentenceTransformer query embedding)
  │        → cosine similarity over pre-built vector DB
  │        → adds {code, term, nel_score} to each annotation
  │
  ▼
 Negation (optional, negation=true only)
  │        dedicated NER model produces NEG/NSCO/UNC/USCO spans
  │        → overlap detection adds {is_negated, is_uncertain, ...}
  │
  ▼
 Post-processing
        merge contiguous same-class entities, deduplicate, sort by offset
      │
      ▼
 Dt4hFormatter.serialize()
        renames fields to CDM v2 names, validates via Pydantic
        wraps into NlpResponse {nlp_output, nlp_service_info}
      │
      ▼
 JSON array response (one NlpResponse per input item)
```

### Key files

| Path | Role |
|---|---|
| `app/__init__.py` | Flask app, endpoints, pipeline cache |
| `app/config.py` | Registry and resource base paths |
| `app/src/pipelines.py` | All pipeline implementations |
| `app/src/format/dt4h.py` | DT4H CDM v2 formatter |
| `app/src/format/data_structures.py` | Pydantic models for CDM v2 schema |
| `app/model_manager/resolver.py` | Single source of truth for resource paths |
| `app/model_manager/registry.yaml` | Model and gazetteer path registry |
| `test_init.py` | Pre-flight pipeline validation |

---

## License

MIT
