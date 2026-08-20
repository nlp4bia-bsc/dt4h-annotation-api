"""Applying linkers across the nested NER result structure.

``ner_results`` is three levels deep — entity type, document, annotation — and
the linker list is aligned with the outermost level. Getting that alignment
wrong would attach codes from one entity type's gazetteer to another's spans.
"""

from __future__ import annotations

import pytest

from app.src.nel.biencoder import biencoder_inference
from app.src.nel.linker import EntityLinker


@pytest.fixture
def linker(gaz_path, model_path, faiss_index_path) -> EntityLinker:
    return EntityLinker(gaz_path=gaz_path, model_path=model_path, index_path=faiss_index_path)


def annotation(span: str, start: int = 0) -> dict:
    return {
        "start": start,
        "end": start + len(span),
        "span": span,
        "ner_class": "DISEASE",
        "ner_score": 0.99,
    }


def test_annotations_gain_code_term_and_score(linker):
    results = [[[annotation("COVID-19")]]]
    linked = biencoder_inference(results, [linker])

    ann = linked[0][0][0]
    assert ann["code"] == "840539006"
    assert ann["term"] == "COVID-19"
    assert ann["nel_score"] == pytest.approx(1.0, abs=1e-4)


def test_original_ner_fields_survive(linker):
    linked = biencoder_inference([[[annotation("varicela", start=7)]]], [linker])
    ann = linked[0][0][0]
    assert ann["start"] == 7
    assert ann["ner_class"] == "DISEASE"
    assert ann["ner_score"] == 0.99


def test_repeated_spans_across_documents_all_get_linked(linker):
    results = [[
        [annotation("COVID-19")],
        [annotation("varicela"), annotation("COVID-19", start=20)],
    ]]
    linked = biencoder_inference(results, [linker])

    flat = [ann for doc in linked[0] for ann in doc]
    assert [a["code"] for a in flat] == ["840539006", "38907003", "840539006"]


def test_documents_without_annotations_are_left_alone(linker):
    linked = biencoder_inference([[[], [annotation("COVID-19")], []]], [linker])
    assert linked[0][0] == []
    assert linked[0][2] == []
    assert linked[0][1][0]["code"] == "840539006"


def test_an_entity_type_with_no_mentions_is_skipped(linker):
    linked = biencoder_inference([[[]], [[annotation("COVID-19")]]], [linker, linker])
    assert linked[0] == [[]]
    assert linked[1][0][0]["code"] == "840539006"


def test_mismatched_linker_count_is_rejected(linker):
    """Silently zipping to the shorter list would drop an entity type's links."""
    with pytest.raises(ValueError, match="one to one"):
        biencoder_inference([[[annotation("COVID-19")]]], [linker, linker])


def test_scores_are_rounded_for_serialisation(linker):
    linked = biencoder_inference([[[annotation("meningitis bacteriana")]]], [linker])
    score = linked[0][0][0]["nel_score"]
    assert score == round(score, 4)
