"""Entity linking orchestration: mentions → best concept per mention.

One ``EntityLinker`` per entity type. It owns its generators, so the FAISS
index and the encoder are opened once when the linker is constructed and reused
for every later request — not reopened per call, as the previous
``biencoder_inference`` did.

With a single generator the fusion step is skipped entirely and the
generator's own ranking and scores are preserved, so enabling extra generators
is the only thing that can change results.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.src.nel.candidates import (
    BM25Generator,
    CandidateGenerator,
    DenseGenerator,
    ExactMatchGenerator,
    TfidfCharNgramGenerator,
)
from app.src.nel.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from app.src.nel.rerank import get_reranker
from app.src.nel.schemas import MatchCandidate, MentionAnnotation

logger = logging.getLogger(__name__)

# Candidates retrieved per mention before fusion. Only meaningful with more
# than one generator; with one, the top hit is all that is ever read.
DEFAULT_TOP_K = 25


def _clamp_confidence(score: float) -> float:
    """Constrain a similarity to the [0, 1] range a confidence field implies."""
    return float(min(1.0, max(0.0, score)))


class EntityLinker:
    """Links mentions of one entity type to gazetteer concepts.

    Parameters
    ----------
    gaz_path, model_path, index_path:
        Resources for the dense generator, resolved by ``LocalResolver``.
        ``model_path`` and ``index_path`` are required only when ``dense`` is
        on; a purely lexical linker needs the gazetteer and nothing else.
    dense:
        The bi-encoder retriever. On by default — it is the method this
        pipeline was built around. Turning it off leaves the lexical
        generators to link on their own, which is cheap and needs no FAISS
        index, but recall then stops at what the gazetteer literally spells.
        At least one generator must remain enabled.
    exact_match, tfidf_char, bm25:
        Additional candidate generators to fuse with the dense one.
        **All off by default, and turning any on changes which codes are
        emitted.** There is no evaluation harness in this repository, so that
        change cannot be measured here — validate against a labelled set
        before enabling any of them in production.

        ``tfidf_char`` and ``bm25`` share one persisted sparse index, built
        lazily on first use; enabling both costs one index, not two.
    lexical_index_path:
        Where that shared sparse index lives. Required when ``tfidf_char`` or
        ``bm25`` is enabled; ``LocalResolver.get_lexical_index_path`` supplies it.
    rerank_model_path:
        Cross-encoder that rescores the shortlist after retrieval and fusion.
        ``None`` disables reranking, which is the default. Supplied by
        ``LocalResolver.get_rerank_path``, which raises when no cross-encoder
        is configured for the language — every language ships that way.

        A reranker also changes the reported confidence: the cross-encoder
        reads mention and candidate jointly, so when one runs it is the
        authority on both the order and the score. See ``rerank.py``.
    top_k:
        Candidates retrieved per generator before fusion, and the size of the
        shortlist handed to the reranker.
    rrf_k:
        RRF damping constant; see ``fusion.py``.
    """

    def __init__(
        self,
        gaz_path: Path,
        model_path: Path | None = None,
        index_path: Path | None = None,
        *,
        dense: bool = True,
        exact_match: bool = False,
        tfidf_char: bool = False,
        bm25: bool = False,
        lexical_index_path: Path | None = None,
        rerank_model_path: Path | None = None,
        reranker: object | None = None,
        top_k: int = DEFAULT_TOP_K,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self.top_k = max(1, top_k)
        self.rrf_k = rrf_k
        # `reranker` is an injection point for tests, so the suite needs no
        # cross-encoder checkpoint.
        self.reranker = reranker
        if self.reranker is None and rerank_model_path is not None:
            self.reranker = get_reranker(rerank_model_path)

        if (tfidf_char or bm25) and lexical_index_path is None:
            raise ValueError(
                "lexical_index_path is required when tfidf_char or bm25 is enabled"
            )
        if dense and (model_path is None or index_path is None):
            raise ValueError(
                "model_path and index_path are required when dense is enabled"
            )
        if not (dense or exact_match or tfidf_char or bm25):
            raise ValueError("at least one candidate generator must be enabled")

        # Construction order is fixed regardless of how the caller listed the
        # generators: it is the tie-break _reportable_score falls back on, so
        # it has to be a property of the class rather than of the call site.
        self.generators: list[CandidateGenerator] = []
        if dense:
            self.generators.append(
                DenseGenerator(gaz_path=gaz_path, model_path=model_path, index_path=index_path)
            )
        if exact_match:
            self.generators.append(ExactMatchGenerator(gaz_path=gaz_path))
        if tfidf_char:
            self.generators.append(
                TfidfCharNgramGenerator(gaz_path=gaz_path, index_path=lexical_index_path)
            )
        if bm25:
            self.generators.append(
                BM25Generator(gaz_path=gaz_path, index_path=lexical_index_path)
            )

        logger.info(
            "EntityLinker ready: generators=%s, reranker=%s, top_k=%d",
            [generator.method for generator in self.generators],
            getattr(self.reranker, "method", None),
            self.top_k,
        )

    @property
    def fuses(self) -> bool:
        """Whether more than one generator's output is being combined."""
        return len(self.generators) > 1

    def _reportable_score(self, fused: MatchCandidate) -> float:
        """Replace an RRF score with something a reader can interpret.

        ``score`` leaves this class as ``nel_score`` and ends up in the CDM's
        ``concept_confidence``.  A raw RRF score cannot go there: it is a
        positional artefact confined to roughly 0.016–0.033 for two
        generators, it shifts whenever a generator is added, and it says
        nothing about how good the match is.

        The fused candidate therefore reports the score of the generator that
        ranked this code **best** — the one most responsible for the decision.
        Reporting the dense score unconditionally would be actively
        misleading: when a lexical generator rescues a typo the bi-encoder
        missed, the bi-encoder's cosine for that code can be near zero or
        negative, which reads as "no confidence" about a link the pipeline is
        in fact confident in.  Ties fall back to generator order, dense first.

        The result is clamped to [0, 1].  Cosine similarity is defined on
        [-1, 1], but a *confidence* below zero has no meaning to a consumer of
        the CDM; a negative similarity is reported as 0.0.

        Scores from different generators are not strictly comparable, which is
        inherent to fusion — it is a rank-based method precisely because the
        scores are not on one scale.  ``metadata`` retains ``rrf_score``,
        ``source_scores`` and ``source_ranks`` so any decision remains
        traceable.
        """
        source_scores = fused.metadata.get("source_scores", {})
        source_ranks = fused.metadata.get("source_ranks", {})
        if not source_scores:
            return _clamp_confidence(fused.score)

        priority = {generator.method: position for position, generator in enumerate(self.generators)}
        best_method = min(
            source_scores,
            key=lambda method: (source_ranks.get(method, 1 << 30), priority.get(method, 1 << 30)),
        )
        return _clamp_confidence(source_scores[best_method])

    def _shortlist(self, mentions: list[MentionAnnotation]) -> list[list[MatchCandidate]]:
        """Retrieve and fuse, returning a ranked shortlist per mention."""
        # One generator with no reranker needs only the top hit; retrieving a
        # deep list would cost more and change nothing. A reranker needs the
        # shortlist to actually reorder.
        k = self.top_k if (self.fuses or self.reranker is not None) else 1
        per_generator = [generator.generate(mentions, k) for generator in self.generators]

        shortlists: list[list[MatchCandidate]] = []
        for index in range(len(mentions)):
            if not self.fuses:
                candidates = per_generator[0][index]
                for candidate in candidates:
                    # Cosine runs to -1, but this may leave as
                    # concept_confidence; same [0, 1] domain as the fused path.
                    candidate.score = _clamp_confidence(candidate.score)
                shortlists.append(candidates)
                continue

            fused = reciprocal_rank_fusion(
                {
                    generator.method: per_generator[position][index]
                    for position, generator in enumerate(self.generators)
                },
                k=self.rrf_k,
                top_k=self.top_k,
            )
            for candidate in fused:
                # Stash the RRF value and swap in an interpretable similarity
                # while the per-source provenance is still attached.
                candidate.metadata["rrf_score"] = candidate.score
                candidate.score = self._reportable_score(candidate)
            shortlists.append(fused)
        return shortlists

    def link(self, mentions: list[MentionAnnotation]) -> list[MatchCandidate | None]:
        """Return the best candidate per mention, or ``None`` where there is none.

        The returned list is aligned with *mentions*.
        """
        if not mentions:
            return []

        shortlists = self._shortlist(mentions)

        if self.reranker is not None:
            # The cross-encoder reads each (mention, term) pair jointly, so it
            # is the authority on the final order and on the reported score.
            shortlists = self.reranker.rerank(mentions, shortlists, top_k=self.top_k)

        return [candidates[0] if candidates else None for candidates in shortlists]

    def link_texts(self, texts: list[str]) -> dict[str, MatchCandidate]:
        """Link bare mention strings, deduplicated.

        Repeated surface forms are linked once. Mentions with no candidate are
        absent from the mapping rather than present with a null value.
        """
        unique = list(dict.fromkeys(texts))
        if not unique:
            return {}

        results = self.link([MentionAnnotation(text=text) for text in unique])
        return {
            text: candidate
            for text, candidate in zip(unique, results)
            if candidate is not None
        }
