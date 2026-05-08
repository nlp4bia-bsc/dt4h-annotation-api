import json
import os
import uuid
from functools import partial
from pathlib import Path

from flask import Flask, request, jsonify
from app.src.pipelines import LookupPipeline, FuzzyMatchPipeline, BM25OkapiPipeline, BiencoderPipeline
from app.src.format import PassthroughFormatter
from typing import Sequence

app = Flask(__name__)
method2pipeline = {
    'lookup': LookupPipeline,
    'levenshtein': partial(FuzzyMatchPipeline, method='levenshtein'),
    'jaro-winkler': partial(FuzzyMatchPipeline, method='jaro_winkler'),
    'token-sort-ratio': partial(FuzzyMatchPipeline, method='token_sort_ratio'),
    'token-set-ratio': partial(FuzzyMatchPipeline, method='token_set_ratio'),
    'bm25': BM25OkapiPipeline,
    'biencoder': partial(BiencoderPipeline, ner_version=2),
}

cdm2formatter = {
    'none': PassthroughFormatter
}

_pipeline_cache: dict = {}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _extract_pipeline_params(data: dict):
    """Validate and extract lang/method/entities/negation from request dict.
    Returns (params_dict, None) on success or (None, error_str) on failure.
    """
    missing = [f for f in ('lang', 'method', 'entities') if f not in data]
    if missing:
        return None, f"Missing required field(s): {', '.join(repr(f) for f in missing)}"

    method = data['method']
    if method not in method2pipeline:
        return None, f"Unknown method '{method}'. Valid: {list(method2pipeline)}"

    entities = data['entities']
    if not isinstance(entities, list) or not entities or not all(isinstance(e, str) for e in entities):
        return None, "'entities' must be a non-empty list of strings."

    negation = data.get('negation', False)
    if negation and method != 'biencoder':
        return None, f"'negation' is only supported with method 'biencoder', got '{method}'."

    return {
        'method': method,
        'lang': data['lang'],
        'entities': data['entities'],
        'negation': negation,
    }, None


def _build_pipeline(method, lang, entities, negation):
    key = (method, lang, frozenset(entities), negation)
    if key not in _pipeline_cache:
        pipeline_cls = method2pipeline[method]
        if method == 'biencoder':
            _pipeline_cache[key] = pipeline_cls(lang=lang, entities=entities, negation=negation)
        else:
            _pipeline_cache[key] = pipeline_cls(lang=lang, entities=entities)
    return _pipeline_cache[key]


def _sanitize_inputs(
        raw_texts: list[str], 
        raw_metadatas: Sequence[dict | None] | None
    ) -> \
    tuple[
        list[str] | None, 
        Sequence[dict | None] | None,
        str | None
    ]:

    """Accept list of str and optional list of metadata and just checks if all the texts are strings and if fills the metadata list if empty.
    Returns (texts, metadatas, None) or (None, None, error_str).
    """
    texts, metadatas = [], []
    for item in zip(raw_texts, raw_metadatas):
        if isinstance(item[0], str):
            texts.append(item[0])
            metadatas.append(item[1])
        else:
            return None, None, 'Each item in "texts" must be a string. Verify your input format.'
    return texts, metadatas, None


def _run_pipeline(pipeline, texts: list, metadatas: Sequence[dict | None]) -> list:
    formatter = cdm2formatter['none']()
    annotations = pipeline.predict(texts=texts)
    return [
        formatter.serialize(text, ann, meta)
        for text, ann, meta in zip(texts, annotations, metadatas)
    ]


def _write_to_dir(results: list[dict], output_dir: Path, filenames: list[str]) -> list:
    """Write one JSON file per result into output_dir. Returns list of written paths."""
    written = []
    for result, fname in zip(results, filenames):
        out_path = output_dir / fname
        with open(out_path, 'w', encoding='utf-8') as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    return "OK", 200


@app.route('/annotate', methods=['POST'])
def annotate():
    """Annotate a single text or a list of texts.

    Request body:
        text       : str                           — single text (mutually exclusive with 'texts')
        texts      : list[str]                     — list of texts (mutually exclusive with 'text')
        metadata   : dict | None                   — optional metadata for single-text mode
        metadatas  : list[dict | None] | None      — optional metadata list for multi-text mode; length must match 'texts'
        lang       : str                           — language code (e.g. "es")
        method     : str                           — pipeline method
        entities   : list[str]                     — non-empty list of entity types to detect
        negation   : bool  (default false)         — negation/uncertainty detection (biencoder only)
        output_dir : str   (optional)              — if set, results are written as JSON files into this directory
    """
    data = request.json
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    params, err = _extract_pipeline_params(data)
    if err:
        return jsonify({"error": err}), 400

    # Accept 'text' (singular) or 'texts' (list)
    single = False
    if 'text' in data:
        if not isinstance(data['text'], str):
            return jsonify({"error": "'text' must be a string"}), 400
        raw_texts = [data['text']]
        raw_metadatas = [data.get('metadata', None)]
        single = True
    elif 'texts' in data:
        raw_texts = data['texts']
        raw_metadatas = data.get('metadatas', [None] * len(raw_texts))
    else:
        return jsonify({"error": "Missing required field: 'text' (single) or 'texts' (list)"}), 400

    if not isinstance(raw_texts, list) or len(raw_texts) == 0:
        return jsonify({"error": "'texts' must be a non-empty list"}), 400
    
    if len(raw_metadatas) != len(raw_texts):
        return jsonify({"error": f"Length mismatch: {len(raw_texts)} texts but {len(raw_metadatas)} metadata entries. Each text must have a corresponding metadata object (can be null)."}), 400

    texts, metadatas, err = _sanitize_inputs(raw_texts, raw_metadatas)
    if err:
        return jsonify({"error": err}), 400

    pipeline = _build_pipeline(**params)
    results = _run_pipeline(pipeline, texts, metadatas)

    if output_dir := data.get('output_dir', None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filenames = [f"{uuid.uuid4().hex}.json" for _ in results]
        written = _write_to_dir(results, output_dir, filenames)
        return jsonify({"output_dir": str(output_dir), "files_written": [str(p) for p in written], "count": len(written)})
    
    return jsonify(results[0] if single else results)


@app.route('/annotate_dir', methods=['POST'])
def annotate_dir():
    """Annotate all .txt files inside a server-side directory.

    Request body:
        input_dir  : str         — absolute path to directory containing .txt files
        lang       : str
        method     : str
        entities   : list[str]   — non-empty list of entity types to detect
        negation   : bool  (default false)  — negation/uncertainty detection (biencoder only)
        output_dir : str  (optional)        — if set, results are written as <stem>.json files into this directory
    """
    data = request.json
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    if 'input_dir' not in data:
        return jsonify({"error": "Missing required field: 'input_dir'"}), 400

    input_dir = data['input_dir']
    if not isinstance(input_dir, str):
        return jsonify({"error": "'input_dir' must be a string"}), 400
    if not os.path.isdir(input_dir):
        return jsonify({"error": f"'input_dir' does not exist or is not a directory: {input_dir}"}), 400

    params, err = _extract_pipeline_params(data)
    if err:
        return jsonify({"error": err}), 400

    txt_files = sorted(Path(input_dir).glob('*.txt'))
    if not txt_files:
        return jsonify({"error": f"No .txt files found in: {input_dir}"}), 400

    texts = [p.read_text(encoding='utf-8') for p in txt_files]
    filenames = [p.name for p in txt_files]
    metadatas = [{"source_file": str(p)} for p in txt_files]

    pipeline = _build_pipeline(**params)
    results = _run_pipeline(pipeline, texts, metadatas)

    if output_dir := data.get('output_dir'):
        if not isinstance(output_dir, str):
            return jsonify({"error": "'output_dir' must be a string"}), 400
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_filenames = [Path(f).stem + '.json' for f in filenames]
        written = _write_to_dir(results, output_dir, out_filenames)
        return jsonify({"output_dir": str(output_dir), "files_written": [str(p) for p in written], "count": len(written)})

    return jsonify(dict(zip(filenames, results)))


if __name__ == '__main__':
    app.run(debug=True)
