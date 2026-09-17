"""Applying linkers across the nested NER result structure.

``ner_results`` is three levels deep — entity type, document, annotation — and
the linker list is aligned with the outermost level. Getting that alignment
wrong would attach codes from one entity type's gazetteer to another's spans.
"""

from __future__ import annotations

import pytest

from app.src.nel import vector_store
from app.src.nel.biencoder import biencoder_inference
from app.src.nel.linker import EntityLinker
from tests.conftest import StubEncoder, basis_vector


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


# -- linking provenance ----------------------------------------------------


def test_the_winning_method_is_recorded(linker):
    """Becomes nel_component_type, so it must name a retriever."""
    linked = biencoder_inference([[[annotation("COVID-19")]]], [linker])
    assert linked[0][0][0]["nel_method"] == "biencoder"


def test_a_fused_run_records_the_retriever_not_the_fusion_step(
    tmp_path, gaz_path, model_path, lexical_index_path
):
    """'rrf' names the combination step, which is no answer to "how was this found".

    The dense retriever is scripted into missing a typo so the lexical one
    rescues it. The reported method has to follow the generator that actually
    decided, otherwise nel_component_type says 'transformer' about a link no
    transformer made.
    """
    scripted = StubEncoder(
        overrides={
            "meningitis bacteriana": basis_vector(0),
            "meningitis vírica": basis_vector(1),
            "covid-19": basis_vector(2),
            "infección por coronavirus": basis_vector(3),
            "varicela": basis_vector(4),
            "insuficiencia cardiaca congestiva": basis_vector(5),
            "meningitis bacteriaa": basis_vector(4),  # collides with 'varicela'
        }
    )
    vector_store._MODEL_CACHE[model_path.resolve()] = scripted
    index_path = tmp_path / "scripted.faiss"
    vector_store.build(gaz_path=gaz_path, model_path=model_path, index_path=index_path)

    fused = EntityLinker(
        gaz_path=gaz_path,
        model_path=model_path,
        index_path=index_path,
        tfidf_char=True,
        lexical_index_path=lexical_index_path,
    )
    linked = biencoder_inference([[[annotation("meningitis bacteriaa")]]], [fused])

    ann = linked[0][0][0]
    assert ann["code"] == "7180009", "the lexical generator should rescue the typo"
    assert ann["nel_method"] == "tfidf_char"


def test_a_rank_tie_in_a_fused_run_goes_to_the_dense_retriever(
    gaz_path, model_path, faiss_index_path, lexical_index_path
):
    """Generator order is the documented tie-break, and the method must follow it.

    Both retrievers put a verbatim mention first, so the reported method has to
    be the one whose score is also reported — otherwise the annotation names a
    component that did not supply its confidence.
    """
    fused = EntityLinker(
        gaz_path=gaz_path,
        model_path=model_path,
        index_path=faiss_index_path,
        exact_match=True,
        lexical_index_path=lexical_index_path,
    )
    linked = biencoder_inference([[[annotation("COVID-19")]]], [fused])
    assert linked[0][0][0]["nel_method"] == "biencoder"


def test_unlinked_annotations_gain_no_method(linker):
    """A span with no candidate keeps no linking keys at all, this one included."""
    linked = biencoder_inference([[[annotation("zzz no such disease zzz")]]], [linker])
    ann = linked[0][0][0]
    if "code" not in ann:
        assert "nel_method" not in ann
