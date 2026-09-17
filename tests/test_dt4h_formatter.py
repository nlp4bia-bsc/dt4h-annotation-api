"""CDM serialisation: what lands in concept_confidence, and what must not.

``concept_confidence`` carries the NEL linking score. A NER-only run has no
such score and emits null rather than falling back to the extraction score,
which measures something else entirely.

Also covers ``concept_class``, where the checkpoint's own label has to be
translated into the CDM vocabulary — and where an *unrecognised* label must
survive untranslated so the out-of-vocabulary warning still names it.
"""

from __future__ import annotations

import pytest

from app.src.format.data_structures import (
    CONCEPT_CLASSES,
    NEL_COMPONENT_TYPES,
    Annotation,
)
from app.src.format.dt4h import (
    CONCEPT_CLASS_MAP,
    CONTROLLED_VOCABULARY_VERSION,
    NEL_COMPONENT_TYPE_MAP,
    NEL_COMPONENT_VERSION,
    Dt4hFormatter,
)

LINKED = {
    "ner_class": "DISEASE",
    "start": 0,
    "end": 8,
    "span": "COVID-19",
    "ner_score": 0.9912,
    "code": "840539006",
    "term": "COVID-19",
    "nel_score": 0.9421,
    "nel_method": "biencoder",
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
    """run_nerl.py produces these; a missing nel_score is not an error."""
    renamed = formatter._rename_annotation(NER_ONLY)
    assert renamed["controlled_vocabulary_concept_identifier"] is None
    assert renamed["controlled_vocabulary_concept_official_term"] is None


# ---------------------------------------------------------------------------
# Linking fields
# ---------------------------------------------------------------------------


def test_dt4h_concept_identifier_is_the_gazetteer_code(formatter):
    renamed = formatter._rename_annotation(LINKED)
    assert renamed["dt4h_concept_identifier"] == "840539006"
    assert renamed["dt4h_concept_identifier"] == renamed["controlled_vocabulary_concept_identifier"]


def test_linked_annotations_carry_the_vocabulary_constants(formatter):
    renamed = formatter._rename_annotation(LINKED)
    assert renamed["controlled_vocabulary_version"] == CONTROLLED_VOCABULARY_VERSION
    assert renamed["nel_component_version"] == NEL_COMPONENT_VERSION


@pytest.mark.parametrize(
    "label, expected_namespace",
    [
        ("DISEASE", "SNOMED CT"),
        ("SYMPTOM", "SNOMED CT"),
        ("PROCEDURE", "SNOMED CT"),
        ("DRUG", "UMLS"),
        ("MEDICATION", "UMLS"),
        ("medicamento", "UMLS"),  # the Spanish checkpoints' own label
    ],
)
def test_namespace_is_umls_for_drugs_and_snomed_otherwise(
    formatter, label, expected_namespace
):
    """Keyed on the *mapped* class, so every alias of 'drug' resolves alike."""
    renamed = formatter._rename_annotation({**LINKED, "ner_class": label})
    assert renamed["controlled_vocabulary_namespace"] == expected_namespace


def test_source_mirrors_the_namespace(formatter):
    """A project decision, not the CDM's own reading of the field.

    ``controlled_vocabulary_source`` is documented as the provenance of the
    term; carrying the terminology name means the validator warns on every
    linked annotation. Pinned here so the warning is never mistaken for a bug.
    """
    for label in ("DISEASE", "DRUG"):
        renamed = formatter._rename_annotation({**LINKED, "ner_class": label})
        assert renamed["controlled_vocabulary_source"] == renamed["controlled_vocabulary_namespace"]


def test_unlinked_annotations_carry_no_vocabulary_fields(formatter):
    """These describe a code. Without one there is nothing to describe."""
    renamed = formatter._rename_annotation(NER_ONLY)
    for cdm_field in (
        "dt4h_concept_identifier",
        "controlled_vocabulary_namespace",
        "controlled_vocabulary_version",
        "controlled_vocabulary_source",
        "nel_component_type",
        "nel_component_version",
    ):
        assert Annotation(**renamed).model_dump()[cdm_field] is None


@pytest.mark.parametrize(
    "method, expected",
    [
        ("biencoder", "transformer"),
        ("cross_encoder", "transformer"),
        ("exact_match", "lexical similarity"),
        ("tfidf_char", "lexical similarity"),
        ("bm25", "lexical similarity"),
        ("something_new", "other"),
    ],
)
def test_nel_component_type_follows_the_method_that_won(formatter, method, expected):
    renamed = formatter._rename_annotation({**LINKED, "nel_method": method})
    assert renamed["nel_component_type"] == expected


def test_every_mapped_component_type_is_a_cdm_value():
    for value in NEL_COMPONENT_TYPE_MAP.values():
        assert value in NEL_COMPONENT_TYPES


def test_a_link_with_no_recorded_method_leaves_the_type_null(formatter):
    """'other' would assert a kind of component; not knowing is not a kind."""
    linked_without_method = {k: v for k, v in LINKED.items() if k != "nel_method"}
    renamed = formatter._rename_annotation(linked_without_method)
    assert Annotation(**renamed).model_dump()["nel_component_type"] is None
    assert renamed["nel_component_version"] == NEL_COMPONENT_VERSION


def test_span_fields_map_across(formatter):
    renamed = formatter._rename_annotation(LINKED)
    assert renamed["concept_class"] == "disorder/disease"
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


# ---------------------------------------------------------------------------
# concept_class mapping
# ---------------------------------------------------------------------------


def test_every_mapped_value_is_a_cdm_concept_class():
    """The map is the only thing standing between model labels and the CDM.

    A typo in a value would produce exactly the warning the map exists to
    silence, so the table is checked against the vocabulary itself.
    """
    assert set(CONCEPT_CLASS_MAP.values()) <= set(CONCEPT_CLASSES)


@pytest.mark.parametrize(
    "label, expected",
    [
        ("DISEASE", "disorder/disease"),
        ("PROCEDURE", "procedure"),
        ("SYMPTOM", "symptom"),
        ("MEDICATION", "medication"),
        # the registry entity type is 'drug'; the CDM value is 'medication'
        ("DRUG", "medication"),
        ("MED", "medication"),
        # the older Spanish checkpoints label in Spanish
        ("ENFERMEDAD", "disorder/disease"),
        ("PROCEDIMIENTO", "procedure"),
        ("SINTOMA", "symptom"),
        ("FÁRMACO", "medication"),
        # a conformant checkpoint round-trips untouched
        ("disorder/disease", "disorder/disease"),
        ("cardiology entity", "cardiology entity"),
        # separators and BIO prefixes are normalised away
        ("CARDIOLOGY_ENTITY", "cardiology entity"),
        ("B-DISEASE", "disorder/disease"),
        ("  disease  ", "disorder/disease"),
    ],
)
def test_labels_map_onto_the_cdm_vocabulary(formatter, label, expected):
    renamed = formatter._rename_annotation({**NER_ONLY, "ner_class": label})
    assert renamed["concept_class"] == expected


def test_an_unknown_label_is_passed_through_not_flattened(formatter):
    """Coercing to 'other' would destroy the warning that names the label."""
    renamed = formatter._rename_annotation({**NER_ONLY, "ner_class": "ANATOMY"})
    assert renamed["concept_class"] == "ANATOMY"
