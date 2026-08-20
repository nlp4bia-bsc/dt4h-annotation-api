"""CDM serialisation: what lands in concept_confidence, and what must not.

``concept_confidence`` carries the NEL linking score. A NER-only run has no
such score and emits null rather than falling back to the extraction score,
which measures something else entirely.
"""

from __future__ import annotations

import pytest

from app.src.format.dt4h import Dt4hFormatter

LINKED = {
    "ner_class": "DISEASE",
    "start": 0,
    "end": 8,
    "span": "COVID-19",
    "ner_score": 0.9912,
    "code": "840539006",
    "term": "COVID-19",
    "nel_score": 0.9421,
}

NER_ONLY = {
    "ner_class": "DISEASE",
    "start": 0,
    "end": 8,
    "span": "COVID-19",
    "ner_score": 0.9912,
}


@pytest.fixture
def formatter() -> Dt4hFormatter:
    return Dt4hFormatter()


def test_concept_confidence_is_the_nel_score(formatter):
    renamed = formatter._rename_annotation(LINKED)
    assert renamed["concept_confidence"] == pytest.approx(0.9421)


def test_concept_confidence_is_not_the_ner_score(formatter):
    renamed = formatter._rename_annotation(LINKED)
    assert renamed["concept_confidence"] != pytest.approx(LINKED["ner_score"])


def test_ner_only_annotations_emit_a_null_confidence(formatter):
    renamed = formatter._rename_annotation(NER_ONLY)
    assert renamed["concept_confidence"] is None


def test_ner_only_annotations_do_not_raise(formatter):
    """run_ner.py produces these; a missing nel_score is not an error."""
    renamed = formatter._rename_annotation(NER_ONLY)
    assert renamed["controlled_vocabulary_concept_identifier"] is None
    assert renamed["controlled_vocabulary_concept_official_term"] is None


def test_span_fields_map_across(formatter):
    renamed = formatter._rename_annotation(LINKED)
    assert renamed["concept_class"] == "DISEASE"
    assert renamed["start_offset"] == 0
    assert renamed["end_offset"] == 8
    assert renamed["concept_mention_string"] == "COVID-19"
    assert renamed["controlled_vocabulary_concept_identifier"] == "840539006"


@pytest.mark.parametrize("missing", ["ner_class", "start", "end", "span"])
def test_a_missing_span_field_still_raises(formatter, missing):
    annotation = {key: value for key, value in LINKED.items() if key != missing}
    with pytest.raises(ValueError, match="Missing expected annotation field"):
        formatter._rename_annotation(annotation)


def test_negation_absent_means_not_assessed_not_negative(formatter):
    """Reporting "no" for an unassessed entity would state a finding never made."""
    renamed = formatter._rename_annotation(LINKED)
    assert "negation" not in renamed


@pytest.mark.parametrize(
    "is_negated, expected", [(True, "yes"), (False, "no")]
)
def test_negation_is_serialised_when_assessed(formatter, is_negated, expected):
    renamed = formatter._rename_annotation(
        {**LINKED, "is_negated": is_negated, "negation_score": 0.77}
    )
    assert renamed["negation"] == expected
    assert renamed["negation_confidence"] == pytest.approx(0.77)
