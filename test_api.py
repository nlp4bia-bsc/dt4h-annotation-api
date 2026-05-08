#!/usr/bin/env python3
"""
API test script.

Usage:
  uv run test_api.py                       # full tests (needs running server + loaded models)
  uv run test_api.py --validation-only     # request validation only (no models needed, fast)
  uv run test_api.py --url http://host:5000
"""

import sys
import argparse
import tempfile
from pathlib import Path
import requests

BASE_URL = "http://localhost:5000"

GREEN = "\033[32m"
RED   = "\033[31m"
RESET = "\033[0m"
BOLD  = "\033[1m"

_passed = _failed = 0

def check(name, condition, detail=""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  {GREEN}PASS{RESET}  {name}")
    else:
        _failed += 1
        print(f"  {RED}FAIL{RESET}  {name}" + (f"  ({detail})" if detail else ""))
    return condition


# ---------------------------------------------------------------------------
# Validation-only tests — no model inference, just HTTP 400 checks
# ---------------------------------------------------------------------------

def test_health():
    print(f"\n{BOLD}Health check{RESET}")
    r = requests.get(f"{BASE_URL}/")
    check("GET / → 200", r.status_code == 200, r.status_code)
    check("body is 'OK'", r.text.strip() == "OK", repr(r.text))


def test_annotate_validation():
    print(f"\n{BOLD}POST /annotate (single text) — validation{RESET}")
    base = {"text": "el paciente tiene cáncer", "lang": "es", "method": "biencoder", "entities": ["disease"]}

    for field in ("text", "lang", "method", "entities"):
        body = {k: v for k, v in base.items() if k != field}
        r = requests.post(f"{BASE_URL}/annotate", json=body)
        check(f"missing '{field}' → 400", r.status_code == 400, r.text)

    r = requests.post(f"{BASE_URL}/annotate", json={**base, "method": "unknown"})
    check("unknown method → 400", r.status_code == 400, r.text)

    r = requests.post(f"{BASE_URL}/annotate", json={**base, "text": 42})
    check("non-string 'text' → 400", r.status_code == 400, r.text)

    r = requests.post(f"{BASE_URL}/annotate", json={**base, "entities": "disease"})
    check("entities as bare string → 400", r.status_code == 400, r.text)

    r = requests.post(f"{BASE_URL}/annotate", json={**base, "entities": []})
    check("empty entities list → 400", r.status_code == 400, r.text)

    r = requests.post(f"{BASE_URL}/annotate", json={**base, "negation": True, "method": "bm25"})
    check("negation=True with non-biencoder → 400", r.status_code == 400, r.text)


    print(f"\n{BOLD}POST /annotate (text list) — validation{RESET}")
    base = {"texts": ["el paciente tiene cáncer"], "lang": "es", "method": "biencoder", "entities": ["disease"]}

    for field in ("texts", "lang", "method", "entities"):
        body = {k: v for k, v in base.items() if k != field}
        r = requests.post(f"{BASE_URL}/annotate", json=body)
        check(f"missing '{field}' → 400", r.status_code == 400, r.text)

    r = requests.post(f"{BASE_URL}/annotate", json={**base, "texts": []})
    check("empty texts list → 400", r.status_code == 400, r.text)

    r = requests.post(f"{BASE_URL}/annotate", json={**base, "texts": [123]})
    check("non-string item in texts → 400", r.status_code == 400, r.text)

    r = requests.post(f"{BASE_URL}/annotate", json={**base, "metadatas": [None, None]})
    check("metadatas length mismatch → 400", r.status_code == 400, r.text)


def test_directory_validation():
    print(f"\n{BOLD}POST /annotate_dir — validation{RESET}")
    base = {"input_dir": "/tmp", "lang": "es", "method": "biencoder", "entities": ["disease"]}

    for field in ("input_dir", "lang", "method", "entities"):
        body = {k: v for k, v in base.items() if k != field}
        r = requests.post(f"{BASE_URL}/annotate_dir", json=body)
        check(f"missing '{field}' → 400", r.status_code == 400, r.text)

    r = requests.post(f"{BASE_URL}/annotate_dir", json={**base, "input_dir": "/nonexistent/xyz"})
    check("nonexistent dir → 400", r.status_code == 400, r.text)

    with tempfile.TemporaryDirectory() as empty_dir:
        r = requests.post(f"{BASE_URL}/annotate_dir", json={**base, "input_dir": empty_dir})
        check("empty dir (no .txt files) → 400", r.status_code == 400, r.text)

    r = requests.post(f"{BASE_URL}/annotate_dir", json={**base, "negation": True, "method": "bm25"})
    check("negation=True with non-biencoder → 400", r.status_code == 400, r.text)


# ---------------------------------------------------------------------------
# Full pipeline tests — require a running server with loaded models
# ---------------------------------------------------------------------------

TEXTS = [
    "el paciente presenta cáncer y dolor de cabeza intenso",
    "diagnóstico: neumonía bilateral con consolidación",
]
PARAMS = {"lang": "es", "method": "biencoder", "entities": ["disease", "symptoms"]}


def test_annotate_full():
    print(f"\n{BOLD}POST /annotate — full pipeline (single text){RESET}")

    # Basic return
    r = requests.post(f"{BASE_URL}/annotate", json={"text": TEXTS[0], **PARAMS})
    check("returns 200", r.status_code == 200, r.text[:200])
    if r.status_code == 200:
        body = r.json()
        check("has 'annotations' key", "annotations" in body)
        check("has 'metadata' key", "metadata" in body)
        check("processing_success is True", body.get("processing_success") is True)
        check("text echoed in metadata", body["metadata"].get("text") == TEXTS[0])

    # With per-text metadata (singular key)
    r = requests.post(f"{BASE_URL}/annotate", json={"text": TEXTS[0], **PARAMS, "metadata": {"patient_id": "42"}})
    check("with metadata → 200", r.status_code == 200, r.text[:200])
    if r.status_code == 200:
        check("metadata echoed", r.json()["metadata"].get("patient_id") == "42")

    # With negation
    r = requests.post(f"{BASE_URL}/annotate", json={"text": TEXTS[0], **PARAMS, "negation": True})
    check("negation=True → 200", r.status_code == 200, r.text[:200])

    # Save to output_dir
    with tempfile.TemporaryDirectory() as tmpdir:
        r = requests.post(f"{BASE_URL}/annotate", json={"text": TEXTS[0], **PARAMS, "output_dir": tmpdir})
        check("output_dir → 200", r.status_code == 200, r.text[:200])
        if r.status_code == 200:
            body = r.json()
            check("response has 'output_dir'", "output_dir" in body)
            check("response has 'files_written'", "files_written" in body)
            check("1 file written", body.get("count") == 1)
            check("file exists on disk", body.get("files_written") and Path(body["files_written"][0]).exists())

    print(f"\n{BOLD}POST /annotate — full pipeline (list of texts){RESET}")

    # Plain list
    r = requests.post(f"{BASE_URL}/annotate", json={"texts": TEXTS, **PARAMS})
    check("returns 200", r.status_code == 200, r.text[:200])
    if r.status_code == 200:
        body = r.json()
        check("returns list of 2", isinstance(body, list) and len(body) == 2)
        check("each item has 'annotations'", all("annotations" in item for item in body))

    # With metadatas list
    metadatas = [{"record_id": "1"}, {"record_id": "2"}]
    r = requests.post(f"{BASE_URL}/annotate", json={"texts": TEXTS, **PARAMS, "metadatas": metadatas})
    check("with metadatas → 200", r.status_code == 200, r.text[:200])
    if r.status_code == 200:
        body = r.json()
        check("metadata echoed for item 1", body[0]["metadata"].get("record_id") == "1")
        check("metadata echoed for item 2", body[1]["metadata"].get("record_id") == "2")

    # Save to output_dir
    with tempfile.TemporaryDirectory() as tmpdir:
        r = requests.post(f"{BASE_URL}/annotate", json={"texts": TEXTS, **PARAMS, "output_dir": tmpdir})
        check("output_dir → 200", r.status_code == 200, r.text[:200])
        if r.status_code == 200:
            body = r.json()
            check("2 files written", body.get("count") == 2)
            saved_files = list(Path(tmpdir).glob("*.json"))
            check("2 json files on disk", len(saved_files) == 2, str(saved_files))


def test_annotate_directory_full():
    print(f"\n{BOLD}POST /annotate_dir — full pipeline{RESET}")

    with tempfile.TemporaryDirectory() as in_dir:
        for i, text in enumerate(TEXTS):
            (Path(in_dir) / f"note_{i}.txt").write_text(text, encoding='utf-8')

        # Return mode — result is a dict keyed by filename
        r = requests.post(f"{BASE_URL}/annotate_dir", json={"input_dir": in_dir, **PARAMS})
        check("returns 200", r.status_code == 200, r.text[:200])
        if r.status_code == 200:
            body = r.json()
            check("returns dict", isinstance(body, dict))
            check("keyed by both filenames", {"note_0.txt", "note_1.txt"} == set(body.keys()))
            check("each value has 'annotations'", all("annotations" in v for v in body.values()))

        # Save to output_dir — filenames match input stems
        with tempfile.TemporaryDirectory() as out_dir:
            r = requests.post(f"{BASE_URL}/annotate_dir", json={
                "input_dir": in_dir, **PARAMS,
                "output_dir": out_dir,
            })
            check("output_dir → 200", r.status_code == 200, r.text[:200])
            if r.status_code == 200:
                body = r.json()
                check("2 files written", body.get("count") == 2)
                saved = list(Path(out_dir).glob("*.json"))
                check("2 json files on disk", len(saved) == 2, str(saved))
                check("filenames match input stems", {f.stem for f in saved} == {"note_0", "note_1"})

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEL API test runner")
    parser.add_argument("--url", default="http://localhost:5000", help="Base URL of the running API server")
    parser.add_argument("--validation-only", action="store_true", help="Only run request-validation tests (no model inference)")
    parser.add_argument("--inference-only", action="store_true", help="Only run inference tests (no request validation)")
    args = parser.parse_args()

    BASE_URL = args.url

    test_health()
    if not args.inference_only:
        test_annotate_validation()
        test_directory_validation()
    else:
        print(f"\n(Skipping validation tests — inference only)")

    if not args.validation_only:
        test_annotate_full()
        test_annotate_directory_full()
    else:
        print(f"\n(Skipping full pipeline tests — validation only)")

    print(f"\n{'='*40}")
    total = _passed + _failed
    color = GREEN if _failed == 0 else RED
    print(f"{color}{_passed}/{total} passed{RESET}" + (f"  ({_failed} failed)" if _failed else ""))
    sys.exit(0 if _failed == 0 else 1)
