"""Reciprocal Rank Fusion for entity-linking candidate lists."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from ..schemas import MatchCandidate


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[MatchCandidate]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[MatchCandidate]:
    """Fuse candidate rankings while retaining source scores and source ranks.

    Args:
        rankings: Mapping from source name to a ranked candidate sequence.
        k: RRF rank constant in ``sum(1 / (k + rank))``.
        top_k: Optional maximum number of fused candidates.

    Returns:
        Candidates ordered by fused score. Duplicate concept codes are merged.
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
            if candidate.code is None:
                continue
            code = str(candidate.code)
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
        key=lambda code: (-fused_scores[code], min(source_ranks[code].values()), code),
    )
    if top_k is not None:
        ordered_codes = ordered_codes[:top_k]

    output: list[MatchCandidate] = []
    for rank, code in enumerate(ordered_codes, 1):
        representative = representatives[code]
        metadata = dict(representative.metadata)
        metadata.update(
            {
                "retrieval_sources": sorted(source_ranks[code]),
                "source_scores": source_scores[code],
                "source_ranks": source_ranks[code],
            }
        )
        output.append(
            MatchCandidate(
                mention_id=representative.mention_id,
                filename=representative.filename,
                text=representative.text,
                label=representative.label,
                code=representative.code,
                candidate_term=representative.candidate_term,
                score=fused_scores[code],
                method="rrf",
                rank=rank,
                metadata=metadata,
            )
        )
    return output
