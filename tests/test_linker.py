"""EntityLinker: generator wiring, fusion, and what reaches concept_confidence.

``nel_score`` becomes the CDM's ``concept_confidence``, so the score rules
tested here are user-visible clinical output, not an internal detail.
"""

from __future__ import annotations

import pytest

from app.src.nel import vector_store
from app.src.nel.candidates import ExactMatchGenerator
from app.src.nel.linker import EntityLinker
from app.src.nel.schemas import MentionAnnotation
from tests.conftest import StubEncoder, basis_vector


# -- exact match -----------------------------------------------------------


def test_exact_match_ignores_case_and_surrounding_space(gaz_path):
    generator = ExactMatchGenerator(gaz_path=gaz_path)
    candidates = generator.generate([MentionAnnotation(text="  covid-19  ")], k=5)[0]
    assert candidates[0].code == "840539006"
    assert candidates[0].rank == 1
    assert candidates[0].score == 1.0


def test_exact_match_returns_nothing_for_an_unlisted_form(gaz_path):
    generator = ExactMatchGenerator(gaz_path=gaz_path)
    assert generator.generate([MentionAnnotation(text="covid nineteen")], k=5)[0] == []


# -- default configuration -------------------------------------------------


def test_default_linker_is_dense_only_and_does_not_fuse(gaz_path, model_path, faiss_index_path):
    linker = EntityLinker(gaz_path=gaz_path, model_path=model_path, index_path=faiss_index_path)
    assert linker.fuses is False
    assert [g.method for g in linker.generators] == ["biencoder"]

    linked = linker.link_texts(["COVID-19"])
    assert linked["COVID-19"].code == "840539006"
    assert linked["COVID-19"].method == "biencoder"


def test_link_texts_deduplicates_repeated_mentions(gaz_path, model_path, faiss_index_path):
    linker = EntityLinker(gaz_path=gaz_path, model_path=model_path, index_path=faiss_index_path)
    linked = linker.link_texts(["COVID-19", "COVID-19", "varicela"])
    assert set(linked) == {"COVID-19", "varicela"}


def test_empty_input_is_handled(gaz_path, model_path, faiss_index_path):
    linker = EntityLinker(gaz_path=gaz_path, model_path=model_path, index_path=faiss_index_path)
    assert linker.link([]) == []
    assert linker.link_texts([]) == {}


def test_lexical_generators_require_their_index_path(gaz_path, model_path, faiss_index_path):
    with pytest.raises(ValueError, match="lexical_index_path"):
        EntityLinker(
            gaz_path=gaz_path,
            model_path=model_path,
            index_path=faiss_index_path,
            tfidf_char=True,
        )


def test_dense_can_be_switched_off_leaving_a_lexical_only_linker(
    gaz_path, lexical_index_path
):
    """A lexical-only linker must not need the encoder or the FAISS index.

    That is the whole point of offering the methods separately: no model_path,
    no index_path, and no vector DB has to exist on disk for it to link.
    """
    linker = EntityLinker(
        gaz_path=gaz_path,
        dense=False,
        exact_match=True,
        tfidf_char=True,
        lexical_index_path=lexical_index_path,
    )
    assert [g.method for g in linker.generators] == ["exact_match", "tfidf_char"]

    linked = linker.link_texts(["COVID-19"])
    assert linked["COVID-19"].code == "840539006"


def test_dense_requires_its_model_and_index_paths(gaz_path):
    with pytest.raises(ValueError, match="model_path and index_path"):
        EntityLinker(gaz_path=gaz_path)


def test_disabling_every_generator_is_rejected(gaz_path):
    with pytest.raises(ValueError, match="at least one candidate generator"):
        EntityLinker(gaz_path=gaz_path, dense=False)


def test_every_generator_can_be_enabled_together(
    gaz_path, model_path, faiss_index_path, lexical_index_path
):
    linker = EntityLinker(
        gaz_path=gaz_path,
        model_path=model_path,
        index_path=faiss_index_path,
        exact_match=True,
        tfidf_char=True,
        bm25=True,
        lexical_index_path=lexical_index_path,
    )
    assert [g.method for g in linker.generators] == [
        "biencoder", "exact_match", "tfidf_char", "bm25",
    ]
    assert lexical_index_path.exists(), "the sparse index should be built on first use"


# -- score semantics -------------------------------------------------------


def test_fused_score_is_a_similarity_not_the_rrf_artefact(
    gaz_path, model_path, faiss_index_path, lexical_index_path
):
    """RRF scores sit near 0.02 regardless of match quality.

    Emitting one as concept_confidence would report a near-zero confidence for
    a perfect match, so the linker substitutes a real similarity and keeps the
    RRF value only for debugging.
    """
    linker = EntityLinker(
        gaz_path=gaz_path,
        model_path=model_path,
        index_path=faiss_index_path,
        exact_match=True,
        lexical_index_path=lexical_index_path,
    )
    candidate = linker.link_texts(["COVID-19"])["COVID-19"]

    assert candidate.score == pytest.approx(1.0, abs=1e-4)
    assert candidate.metadata["rrf_score"] < 0.05
    assert candidate.score != candidate.metadata["rrf_score"]


def test_confidence_comes_from_the_generator_that_ranked_the_winner_best(
    tmp_path, gaz_path, model_path, lexical_index_path
):
    """A lexical rescue must not be reported with the dense retriever's score.

    The stub encoder is scripted so the bi-encoder puts the typo *far* from the
    right concept — the situation fusion exists to fix. Reporting the dense
    cosine there would attach a near-zero confidence to a correct link.
    """
    # 'varicela' and the typo share a direction; the correct concept is
    # orthogonal to the query, so dense retrieval cannot rank it first.
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

    dense_only = EntityLinker(gaz_path=gaz_path, model_path=model_path, index_path=index_path)
    assert dense_only.link_texts(["meningitis bacteriaa"])["meningitis bacteriaa"].code == "38907003"

    fused = EntityLinker(
        gaz_path=gaz_path,
        model_path=model_path,
        index_path=index_path,
        tfidf_char=True,
        lexical_index_path=lexical_index_path,
    )
    candidate = fused.link_texts(["meningitis bacteriaa"])["meningitis bacteriaa"]

    assert candidate.code == "7180009", "the lexical generator should rescue the typo"
    assert candidate.score > 0.5, "confidence must reflect the generator that decided"
    assert candidate.metadata["source_ranks"]["tfidf_char"] == 1


def test_confidence_is_clamped_to_the_unit_interval(
    tmp_path, gaz_path, model_path
):
    """Cosine runs to -1; a negative *confidence* is meaningless in the CDM."""
    opposed = StubEncoder(
        overrides={
            "meningitis bacteriana": basis_vector(0),
            "meningitis vírica": basis_vector(0),
            "covid-19": basis_vector(0),
            "infección por coronavirus": basis_vector(0),
            "varicela": basis_vector(0),
            "insuficiencia cardiaca congestiva": basis_vector(0),
            "anything": -basis_vector(0),
        }
    )
    vector_store._MODEL_CACHE[model_path.resolve()] = opposed
    index_path = tmp_path / "opposed.faiss"
    vector_store.build(gaz_path=gaz_path, model_path=model_path, index_path=index_path)

    linker = EntityLinker(gaz_path=gaz_path, model_path=model_path, index_path=index_path)
    candidate = linker.link_texts(["anything"])["anything"]

    assert candidate.score == 0.0, "a -1.0 cosine must not surface as a negative confidence"


def test_single_generator_preserves_the_dense_ranking(
    gaz_path, model_path, faiss_index_path, encoder
):
    """With one generator nothing is fused, so results cannot drift."""
    linker = EntityLinker(gaz_path=gaz_path, model_path=model_path, index_path=faiss_index_path)
    store = vector_store.load(faiss_index_path, gaz_path, model_path)

    direct = store.search(["varicela"], encoder=encoder, k=1)[0][0]
    linked = linker.link_texts(["varicela"])["varicela"]

    assert linked.code == direct.code
    assert linked.score == pytest.approx(max(0.0, direct.score), abs=1e-6)
