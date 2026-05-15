import threading
from datetime import datetime, timezone
from functools import partial

from flask import Flask, request, jsonify
from app.src.pipelines import LookupPipeline, FuzzyMatchPipeline, BM25OkapiPipeline, BiencoderPipeline
from app.src.format import PassthroughFormatter, Dt4hFormatter
from app.model_manager import ModelManager

app = Flask(__name__)
app.json.sort_keys = False

method2pipeline = {
    'lookup': LookupPipeline,
    'levenshtein': partial(FuzzyMatchPipeline, method='levenshtein'),
    'jaro-winkler': partial(FuzzyMatchPipeline, method='jaro_winkler'),
    'token-sort-ratio': partial(FuzzyMatchPipeline, method='token_sort_ratio'),
    'token-set-ratio': partial(FuzzyMatchPipeline, method='token_set_ratio'),
    'bm25': BM25OkapiPipeline,
    'biencoder': partial(BiencoderPipeline, ner_version=2),
}

_pipeline_cache: dict = {}
_sync_lock = threading.Lock()


def _describe_resource(r) -> dict:
    return {
        "resource": r["resource"],
        "lang": r["lang"],
        "task": r["task"],
        "repo_id": r["repo_id"],
    }


@app.route("/", methods=["GET"])
def health():
    return "OK", 200


@app.route('/process_bulk', methods=['POST'])
def process_bulk():
    """Process a batch of clinical texts from a CogStack/NiFi pipeline.

    Query params:
        language  : str           — language code (e.g. "es")
        entities  : str           — comma-separated entity types (e.g. "disease,symptoms")
        negation  : str (optional) — "true"/"false", default "false"

    Request body:
        {
          "content": [
            {
              "id":     str,
              "text":   str,
              "footer": dict  (all fields optional)
            },
            ...
          ]
        }
    """
    # --- Query param parsing ---
    language = request.args.get('language')
    if not language:
        return jsonify({"error": "Missing required query param: 'language'"}), 400

    entities_raw = request.args.get('entities')
    if not entities_raw:
        return jsonify({"error": "Missing required query param: 'entities'"}), 400
    entities = [e.strip() for e in entities_raw.split(',') if e.strip()]
    if not entities:
        return jsonify({"error": "'entities' must contain at least one entity type"}), 400

    negation = request.args.get('negation', 'false').lower() == 'true'

    # --- Body parsing ---
    data = request.json
    if not isinstance(data, dict) or 'content' not in data:
        return jsonify({"error": "Request body must be a JSON object with a 'content' key"}), 400

    content = data['content']
    if not isinstance(content, list) or len(content) == 0:
        return jsonify({"error": "'content' must be a non-empty list"}), 400

    texts, footers = [], []
    for i, item in enumerate(content):
        text = item.get('text')
        if not isinstance(text, str) or not text:
            return jsonify({"error": f"Item {i}: 'text' must be a non-empty string"}), 400
        texts.append(text)
        footers.append(item.get('footer') or {})

    # --- Pipeline (hardcoded biencoder, cached) ---
    method = 'biencoder'
    key = (method, language, frozenset(entities), negation)
    if key not in _pipeline_cache:
        _pipeline_cache[key] = method2pipeline[method](lang=language, entities=entities, negation=negation)
    pipeline = _pipeline_cache[key]

    # --- Inference + formatting ---
    formatter = Dt4hFormatter()
    annotations = pipeline.predict(texts=texts)
    results = [
        formatter.serialize(text, ann, footer)
        for text, ann, footer in zip(texts, annotations, footers)
    ]
    return jsonify(results)


@app.route('/sync_models', methods=['POST'])
def sync_models():
    """Download any pending resources from the registry and clear the pipeline cache.

    Returns 200 (no_op or done), 206 (partial_error), 409 (already_running), or 500 (error).
    """
    acquired = _sync_lock.acquire(blocking=False)
    if not acquired:
        return jsonify({"status": "already_running", "message": "Sync already in progress."}), 409

    started_at = datetime.now(timezone.utc).isoformat()
    try:
        manager = ModelManager()
        pending_before = [_describe_resource(r) for r in manager.find_pending_resources()]

        if not pending_before:
            return jsonify({
                "status": "no_op",
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "pending_before": [],
                "completed": [],
                "errors": [],
                "cache_cleared": False,
            }), 200

        manager.sanitize()

        pending_after_keys = {
            (r["resource"], r["lang"], r["task"])
            for r in manager.find_pending_resources()
        }
        completed = [r for r in pending_before if (r["resource"], r["lang"], r["task"]) not in pending_after_keys]
        errors    = [r for r in pending_before if (r["resource"], r["lang"], r["task"]) in pending_after_keys]

        cache_cleared = False
        if not errors:
            _pipeline_cache.clear()
            cache_cleared = True

        return jsonify({
            "status": "done" if not errors else "partial_error",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "pending_before": pending_before,
            "completed": completed,
            "errors": errors,
            "cache_cleared": cache_cleared,
        }), 200 if not errors else 206

    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": str(exc),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }), 500

    finally:
        _sync_lock.release()


if __name__ == '__main__':
    app.run(debug=True)
