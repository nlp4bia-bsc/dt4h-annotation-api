"""Cross-encoder reranking.

No checkpoint is required: the reranker takes an injected model, and these
tests supply a scripted stand-in. That keeps the whole path — pair batching,
score normalisation, reordering, provenance, linker integration — under test
while a real cross-encoder is still unconfigured in the registry.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.src.nel import rerank
from app.src.nel.linker import EntityLinker
from app.src.nel.rerank import CrossEncoderReranker, normalise_scores
from app.src.nel.schemas import MatchCandidate, MentionAnnotation


class StubCrossEncoder:
    """Scores pairs from a lookup table, recording the batches it was given."""

    def __init__(self, table: dict[tuple[str, str], float], default: float = 0.0) -> None:
        self.table = table
        self.default = default
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, **kwargs):
        self.calls.append(list(pairs))
        return np.array(
            [self.table.get((m, t), self.default) for m, t in pairs], dtype=np.float32
        )


def candidate(code: str, term: str, score: float, rank: int, method: str = "biencoder"):
    return MatchCandidate(code=code, term=term, score=score, method=method, rank=rank)


@pytest.fixture(autouse=True)
def _clear_reranker_cache():
    rerank.clear_cache()
    yield
    rerank.clear_cache()


# -- score normalisation ---------------------------------------------------


def test_sigmoid_outputs_pass_through_untouched():
    scores = np.array([0.0, 0.3, 1.0])
    assert normalise_scores(scores) == pytest.approx(scores)


def test_raw_logits_are_squashed_into_the_unit_interval():
    normalised = normalise_scores(np.array([-8.0, 0.0, 8.0]))
    assert normalised.min() >= 0.0 and normalised.max() <= 1.0
    assert normalised[1] == pytest.approx(0.5)


def test_normalisation_is_monotonic_so_ranking_never_changes():
    raw = np.array([-5.0, -1.0, 0.0, 2.0, 7.0])
    normalised = normalise_scores(raw)
    assert list(np.argsort(raw)) == list(np.argsort(normalised))


def test_large_logits_do_not_overflow():
    normalised = normalise_scores(np.array([-800.0, 800.0]))
    assert np.all(np.isfinite(normalised))
    assert normalised[0] == pytest.approx(0.0)
    assert normalised[1] == pytest.approx(1.0)


def test_empty_scores():
    assert normalise_scores(np.array([])).size == 0


# -- reranking -------------------------------------------------------------


def test_reranking_reorders_by_cross_encoder_score():
    """The point of a cross-encoder: overturn the retriever's order."""
    model = StubCrossEncoder({("covid", "COVID-19"): 0.95, ("covid", "varicela"): 0.10})
    reranker = CrossEncoderReranker(model_path="unused", model=model)

    # The retriever put the wrong concept first.
    shortlist = [[candidate("38907003", "varicela", 0.8, 1),
                  candidate("840539006", "COVID-19", 0.7, 2)]]
    result = reranker.rerank([MentionAnnotation(text="covid")], shortlist)[0]

    assert [c.code for c in result] == ["840539006", "38907003"]
    assert [c.rank for c in result] == [1, 2]
    assert result[0].score == pytest.approx(0.95)
    assert result[0].method == "cross_encoder"


def test_prior_ranking_is_preserved_in_metadata():
    model = StubCrossEncoder({("covid", "COVID-19"): 0.95})
    reranker = CrossEncoderReranker(model_path="unused", model=model)

    shortlist = [[candidate("840539006", "COVID-19", 0.71, 3, method="rrf")]]
    result = reranker.rerank([MentionAnnotation(text="covid")], shortlist)[0]

    assert result[0].metadata["pre_rerank_method"] == "rrf"
    assert result[0].metadata["pre_rerank_score"] == pytest.approx(0.71)
    assert result[0].metadata["pre_rerank_rank"] == 3
    assert result[0].metadata["cross_encoder_raw"] == pytest.approx(0.95)


def test_all_mentions_are_scored_in_one_batched_call():
    """Per-mention calls would make reranking far more expensive than it is."""
    model = StubCrossEncoder({})
    reranker = CrossEncoderReranker(model_path="unused", model=model)

    mentions = [MentionAnnotation(text="a"), MentionAnnotation(text="b")]
    shortlist = [
        [candidate("1", "x", 0.5, 1), candidate("2", "y", 0.4, 2)],
        [candidate("3", "z", 0.6, 1)],
    ]
    reranker.rerank(mentions, shortlist)

    assert len(model.calls) == 1
    assert model.calls[0] == [("a", "x"), ("a", "y"), ("b", "z")]


def test_scores_are_not_mixed_between_mentions():
    """The flat pair list must be sliced back to the right mention."""
    model = StubCrossEncoder({("a", "x"): 0.1, ("a", "y"): 0.9, ("b", "z"): 0.2})
    reranker = CrossEncoderReranker(model_path="unused", model=model)

    result = reranker.rerank(
        [MentionAnnotation(text="a"), MentionAnnotation(text="b")],
        [
            [candidate("1", "x", 0.5, 1), candidate("2", "y", 0.4, 2)],
            [candidate("3", "z", 0.6, 1)],
        ],
    )
    assert [c.code for c in result[0]] == ["2", "1"]
    assert result[1][0].score == pytest.approx(0.2)


def test_a_mention_with_no_candidates_stays_empty():
    model = StubCrossEncoder({("b", "z"): 0.5})
    reranker = CrossEncoderReranker(model_path="unused", model=model)

    result = reranker.rerank(
        [MentionAnnotation(text="a"), MentionAnnotation(text="b")],
        [[], [candidate("3", "z", 0.6, 1)]],
    )
    assert result[0] == []
    assert result[1][0].code == "3"


def test_no_candidates_at_all_skips_the_model():
    model = StubCrossEncoder({})
    reranker = CrossEncoderReranker(model_path="unused", model=model)
    result = reranker.rerank([MentionAnnotation(text="a")], [[]])
    assert result == [[]]
    assert model.calls == []


def test_top_k_truncates_the_reranked_list():
    model = StubCrossEncoder({})
    reranker = CrossEncoderReranker(model_path="unused", model=model)
    shortlist = [[candidate(str(i), f"t{i}", 0.5, i) for i in range(1, 6)]]
    result = reranker.rerank([MentionAnnotation(text="a")], shortlist, top_k=2)[0]
    assert len(result) == 2


def test_mismatched_lengths_are_rejected():
    reranker = CrossEncoderReranker(model_path="unused", model=StubCrossEncoder({}))
    with pytest.raises(ValueError, match="mentions"):
        reranker.rerank([MentionAnnotation(text="a")], [[], []])


def test_a_model_returning_the_wrong_number_of_scores_is_caught():
    class Broken:
        def predict(self, pairs, **kwargs):
            return np.array([0.5])

    reranker = CrossEncoderReranker(model_path="unused", model=Broken())
    with pytest.raises(ValueError, match="scores for"):
        reranker.rerank(
            [MentionAnnotation(text="a")],
            [[candidate("1", "x", 0.5, 1), candidate("2", "y", 0.4, 2)]],
        )


def test_a_missing_checkpoint_names_the_registry_key(tmp_path):
    with pytest.raises(FileNotFoundError, match="rerank"):
        CrossEncoderReranker(model_path=tmp_path / "absent")


# -- integration with EntityLinker ----------------------------------------


def test_linker_without_a_reranker_is_the_default(gaz_path, model_path, faiss_index_path):
    linker = EntityLinker(gaz_path=gaz_path, model_path=model_path, index_path=faiss_index_path)
    assert linker.reranker is None


def test_reranker_overturns_the_retrieval_result(gaz_path, model_path, faiss_index_path):
    """End to end: dense retrieval decides one code, the cross-encoder another."""
    linker = EntityLinker(
        gaz_path=gaz_path, model_path=model_path, index_path=faiss_index_path
    )
    dense_choice = linker.link_texts(["covid"])["covid"]

    # Score every gazetteer term at 0, except one that is not the dense pick.
    other_term = "varicela" if dense_choice.term != "varicela" else "COVID-19"
    model = StubCrossEncoder({("covid", other_term): 0.99}, default=0.01)

    reranked_linker = EntityLinker(
        gaz_path=gaz_path,
        model_path=model_path,
        index_path=faiss_index_path,
        reranker=CrossEncoderReranker(model_path="unused", model=model),
    )
    result = reranked_linker.link_texts(["covid"])["covid"]

    assert result.term == other_term
    assert result.term != dense_choice.term
    assert result.method == "cross_encoder"
    assert result.score == pytest.approx(0.99)


def test_a_reranker_makes_a_single_generator_fetch_a_shortlist(
    gaz_path, model_path, faiss_index_path
):
    """With one generator and no reranker only the top hit is retrieved.

    A reranker needs more than one candidate or it has nothing to reorder.
    """
    model = StubCrossEncoder({})
    linker = EntityLinker(
        gaz_path=gaz_path,
        model_path=model_path,
        index_path=faiss_index_path,
        reranker=CrossEncoderReranker(model_path="unused", model=model),
    )
    linker.link([MentionAnnotation(text="covid")])

    assert len(model.calls[0]) > 1, "the reranker was handed a single candidate"


def test_reranked_confidence_stays_in_the_unit_interval(gaz_path, model_path, faiss_index_path):
    """Raw logits must not reach concept_confidence unsquashed."""
    model = StubCrossEncoder({}, default=12.0)
    linker = EntityLinker(
        gaz_path=gaz_path,
        model_path=model_path,
        index_path=faiss_index_path,
        reranker=CrossEncoderReranker(model_path="unused", model=model),
    )
    result = linker.link_texts(["covid"])["covid"]
    assert 0.0 <= result.score <= 1.0
    assert result.metadata["cross_encoder_raw"] == pytest.approx(12.0)
