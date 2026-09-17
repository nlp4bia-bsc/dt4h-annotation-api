"""Sparse lexical index: persistence, rebuild-on-staleness, and both scorers.

The persistence rule here differs from the FAISS store: because these are
pickled scikit-learn estimators, a mismatch *rebuilds* rather than raising.
The invariant is the same either way — a stale artifact is never used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sklearn

from app.src.nel import lexical_index
from app.src.nel.candidates import BM25Generator, TfidfCharNgramGenerator
from app.src.nel.schemas import MentionAnnotation


def manifest_of(index_path: Path) -> dict:
    return json.loads(
        lexical_index._index_manifest_path(index_path).read_text(encoding="utf-8")
    )


def test_build_persists_index_and_manifest(gaz_path, lexical_index_path):
    lexical_index.build(gaz_path, lexical_index_path)
    assert lexical_index_path.exists()

    manifest = manifest_of(lexical_index_path)
    assert manifest["n_rows"] == 6
    assert manifest["sklearn_version"] == sklearn.__version__
    assert manifest["char_ngram_range"] == [3, 5]


def test_a_fresh_index_is_reused_not_rebuilt(gaz_path, lexical_index_path):
    lexical_index.build(gaz_path, lexical_index_path)
    lexical_index.clear_cache()
    assert lexical_index._why_not_reusable(lexical_index_path, gaz_path) is None


def test_load_or_build_caches_by_path(gaz_path, lexical_index_path):
    first = lexical_index.load_or_build(gaz_path, lexical_index_path)
    second = lexical_index.load_or_build(gaz_path, lexical_index_path)
    assert first is second


def test_absent_index_is_built_on_first_use(gaz_path, lexical_index_path):
    assert not lexical_index_path.exists()
    index = lexical_index.load_or_build(gaz_path, lexical_index_path)
    assert lexical_index_path.exists()
    assert index.n_rows == 6


@pytest.mark.parametrize(
    "field, value, expected",
    [
        ("sklearn_version", "0.0.0-fake", "scikit-learn"),
        ("scipy_version", "0.0.0-fake", "scipy"),
        ("manifest_version", 999, "manifest version"),
        ("char_ngram_range", [2, 4], "n-gram range"),
    ],
)
def test_manifest_mismatch_forces_a_rebuild(gaz_path, lexical_index_path, field, value, expected):
    lexical_index.build(gaz_path, lexical_index_path)
    manifest_path = lexical_index._index_manifest_path(lexical_index_path)
    manifest_path.write_text(json.dumps({**manifest_of(lexical_index_path), field: value}))

    reason = lexical_index._why_not_reusable(lexical_index_path, gaz_path)
    assert reason is not None and expected in reason


def test_changed_gazetteer_rebuilds_rather_than_returning_stale_rows(gaz_path, lexical_index_path):
    lexical_index.build(gaz_path, lexical_index_path)
    gaz_path.write_text(
        gaz_path.read_text(encoding="utf-8") + "gripe\t6142004\n", encoding="utf-8"
    )
    lexical_index.clear_cache()

    assert lexical_index._why_not_reusable(lexical_index_path, gaz_path) == "gazetteer has changed"
    rebuilt = lexical_index.load_or_build(gaz_path, lexical_index_path)
    assert rebuilt.n_rows == 7


def test_corrupt_pickle_rebuilds_instead_of_raising(gaz_path, lexical_index_path):
    lexical_index.build(gaz_path, lexical_index_path)
    lexical_index_path.write_bytes(b"not a pickle")
    lexical_index.clear_cache()

    index = lexical_index.load_or_build(gaz_path, lexical_index_path)
    assert index.n_rows == 6


def test_missing_manifest_rebuilds(gaz_path, lexical_index_path):
    lexical_index.build(gaz_path, lexical_index_path)
    lexical_index._index_manifest_path(lexical_index_path).unlink()
    lexical_index.clear_cache()
    assert lexical_index._why_not_reusable(lexical_index_path, gaz_path) == "no manifest"


# -- scoring ---------------------------------------------------------------


def test_tfidf_recovers_a_typo_the_exact_form_would_miss(gaz_path, lexical_index_path):
    generator = TfidfCharNgramGenerator(gaz_path=gaz_path, index_path=lexical_index_path)
    candidates = generator.generate([MentionAnnotation(text="meningitis bacteriaa")], k=3)[0]
    assert candidates[0].code == "7180009"
    assert candidates[0].method == "tfidf_char"


def test_tfidf_is_case_and_spacing_insensitive(gaz_path, lexical_index_path):
    generator = TfidfCharNgramGenerator(gaz_path=gaz_path, index_path=lexical_index_path)
    candidates = generator.generate([MentionAnnotation(text="  COVID-19  ")], k=1)[0]
    assert candidates[0].code == "840539006"


def test_candidates_are_deduplicated_by_code(gaz_path, lexical_index_path):
    """Two rows share 7180009; a candidate list must not offer it twice."""
    generator = TfidfCharNgramGenerator(gaz_path=gaz_path, index_path=lexical_index_path)
    candidates = generator.generate([MentionAnnotation(text="meningitis")], k=5)[0]
    codes = [candidate.code for candidate in candidates]
    assert len(codes) == len(set(codes))


def test_ranks_are_one_based_and_ordered(gaz_path, lexical_index_path):
    generator = TfidfCharNgramGenerator(gaz_path=gaz_path, index_path=lexical_index_path)
    candidates = generator.generate([MentionAnnotation(text="meningitis")], k=5)[0]
    assert [c.rank for c in candidates] == list(range(1, len(candidates) + 1))
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_bm25_scores_word_overlap(gaz_path, lexical_index_path):
    generator = BM25Generator(gaz_path=gaz_path, index_path=lexical_index_path)
    candidates = generator.generate([MentionAnnotation(text="meningitis bacteriana")], k=3)[0]
    assert candidates[0].code == "7180009"
    assert candidates[0].method == "bm25"


def test_a_query_sharing_nothing_returns_no_candidates(gaz_path, lexical_index_path):
    generator = BM25Generator(gaz_path=gaz_path, index_path=lexical_index_path)
    assert generator.generate([MentionAnnotation(text="zzzz nothing matches")], k=3)[0] == []


def test_both_generators_share_one_built_index(gaz_path, lexical_index_path):
    tfidf = TfidfCharNgramGenerator(gaz_path=gaz_path, index_path=lexical_index_path)
    bm25 = BM25Generator(gaz_path=gaz_path, index_path=lexical_index_path)
    assert tfidf.index is bm25.index


def test_empty_mention_list_returns_empty(gaz_path, lexical_index_path):
    generator = TfidfCharNgramGenerator(gaz_path=gaz_path, index_path=lexical_index_path)
    assert generator.generate([], k=5) == []
