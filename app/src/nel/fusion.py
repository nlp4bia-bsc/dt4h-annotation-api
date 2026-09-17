"""Reciprocal Rank Fusion over candidate lists from several generators.

Ported from the reference implementation's ``ensembling/rrf.py``, adapted to
this package's ``MatchCandidate``.

RRF combines rankings by position, never by score::

    fused(code) = sum over generators of  1 / (k + rank_in_that_generator)

That matters here because the generators' scores are not on one scale: the
bi-encoder emits a cosine similarity in [-1, 1] and exact match emits a
constant 1.0. Averaging those would be meaningless; comparing ranks is not.

``k`` damps the influence of top positions. At the conventional default of 60,
rank 1 contributes 1/61 and rank 2 contributes 1/62 — close enough that a
concept ranked reasonably by several generators outranks one ranked first by
a single generator. Lower ``k`` sharpens the advantage of first place.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from app.src.nel.schemas import MatchCandidate

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[MatchCandidate]],
    *,
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[MatchCandidate]:
    """Fuse per-generator candidate lists for a single mention.

    Parameters
    ----------
    rankings:
        Generator name → its ranked candidates for one mention.
    k:
        RRF damping constant. Must be non-negative.
    top_k:
        Optional cap on the number of fused candidates returned.

    Returns
    -------
    list[MatchCandidate]
        Candidates ordered by fused score, duplicate codes merged, ``method``
        set to ``"rrf"`` and ``rank`` renumbered from 1. Each carries its
        per-generator scores and ranks in ``metadata`` so a decision stays
        traceable to the generators that produced it.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if not rankings:
        return []

    fused_scores: dict[str, float] = defaultdict(float)
    representatives: dict[str, MatchCandidate] = {}
    source_scores: dict[str, dict[str, float]] = defaultdict(dict)
    source_ranks: dict[str, dict[str, int]] = defaultdict(dict)

    for source, candidates in rankings.items():
        seen: set[str] = set()
        for fallback_rank, candidate in enumerate(candidates, 1):
            code = str(candidate.code)
            # A generator that returns one code twice must not be able to vote
            # for it twice; only its best position counts.
            if code in seen:
                continue
            seen.add(code)

            rank = int(candidate.rank or fallback_rank)
            fused_scores[code] += 1.0 / (k + rank)
            source_scores[code][source] = float(candidate.score)
            source_ranks[code][source] = rank
            representatives.setdefault(code, candidate)

    ordered_codes = sorted(
        fused_scores,
        # Ties break on best rank achieved in any generator, then on the code
        # itself, so the output is deterministic across runs.
        key=lambda code: (-fused_scores[code], min(source_ranks[code].values()), code),
    )
    if top_k is not None:
        ordered_codes = ordered_codes[:top_k]

    fused: list[MatchCandidate] = []
    for rank, code in enumerate(ordered_codes, 1):
        representative = representatives[code]
        fused.append(
            MatchCandidate(
                code=representative.code,
                term=representative.term,
                score=fused_scores[code],
                method="rrf",
                rank=rank,
                metadata={
                    **representative.metadata,
                    "sources": sorted(source_ranks[code]),
                    "source_scores": source_scores[code],
                    "source_ranks": source_ranks[code],
                },
            )
        )
    return fused
