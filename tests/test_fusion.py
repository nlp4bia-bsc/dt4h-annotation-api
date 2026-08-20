"""Reciprocal rank fusion: the maths, in isolation from any retriever."""

from __future__ import annotations

import pytest

from app.src.nel.fusion import reciprocal_rank_fusion
from app.src.nel.schemas import MatchCandidate


def candidate(code: str, rank: int, score: float = 0.5, method: str = "g") -> MatchCandidate:
    return MatchCandidate(code=code, term=code.lower(), score=score, method=method, rank=rank)


def test_agreement_beats_a_single_first_place():
    """The property RRF exists for: consensus outranks one confident source."""
    first = [candidate("A", 1), candidate("B", 2)]
    second = [candidate("C", 1), candidate("B", 2)]

    fused = reciprocal_rank_fusion({"g1": first, "g2": second}, k=60)

    assert fused[0].code == "B"
    assert fused[0].score == pytest.approx(2 / 62)


def test_a_source_cannot_vote_twice_for_one_code():
    duplicated = [candidate("A", 1), candidate("A", 2)]
    fused = reciprocal_rank_fusion({"g1": duplicated}, k=60)
    assert len(fused) == 1
    assert fused[0].score == pytest.approx(1 / 61)


def test_output_is_renumbered_and_marked_as_fused():
    fused = reciprocal_rank_fusion(
        {"g1": [candidate("A", 1), candidate("B", 2)], "g2": [candidate("B", 1)]}, k=60
    )
    assert [c.rank for c in fused] == [1, 2]
    assert {c.method for c in fused} == {"rrf"}


def test_provenance_is_retained_for_every_code():
    fused = reciprocal_rank_fusion(
        {
            "dense": [candidate("A", 1, score=0.91, method="dense")],
            "lexical": [candidate("A", 3, score=0.42, method="lexical")],
        },
        k=60,
    )
    metadata = fused[0].metadata
    assert metadata["sources"] == ["dense", "lexical"]
    assert metadata["source_ranks"] == {"dense": 1, "lexical": 3}
    assert metadata["source_scores"] == {"dense": pytest.approx(0.91), "lexical": pytest.approx(0.42)}


def test_ties_break_deterministically_on_best_rank_then_code():
    fused = reciprocal_rank_fusion(
        {"g1": [candidate("Z", 1)], "g2": [candidate("A", 1)]}, k=60
    )
    # Equal fused scores and equal best ranks, so the code decides.
    assert [c.code for c in fused] == ["A", "Z"]


def test_top_k_caps_the_output():
    fused = reciprocal_rank_fusion(
        {"g1": [candidate(c, i) for i, c in enumerate("ABCDE", 1)]}, k=60, top_k=2
    )
    assert len(fused) == 2


def test_a_smaller_k_sharpens_the_advantage_of_first_place():
    rankings = {"g1": [candidate("A", 1)], "g2": [candidate("B", 2)]}
    sharp = reciprocal_rank_fusion(rankings, k=1)
    flat = reciprocal_rank_fusion(rankings, k=1000)
    assert sharp[0].score / sharp[1].score > flat[0].score / flat[1].score


def test_empty_rankings_and_empty_lists():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"g1": []}) == []


def test_negative_k_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion({"g1": [candidate("A", 1)]}, k=-1)


def test_missing_rank_falls_back_to_list_position():
    unranked = [
        MatchCandidate(code="A", term="a", score=0.9, method="g", rank=None),
        MatchCandidate(code="B", term="b", score=0.8, method="g", rank=None),
    ]
    fused = reciprocal_rank_fusion({"g1": unranked}, k=60)
    assert [c.code for c in fused] == ["A", "B"]
    assert fused[0].score == pytest.approx(1 / 61)
