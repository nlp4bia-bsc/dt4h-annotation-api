"""Cross-encoder reranking of NEL candidates.

Where a bi-encoder embeds mention and concept *independently* and compares
vectors, a cross-encoder reads the pair together and scores it directly. That
lets it weigh interactions a single vector cannot represent — a negation, a
laterality, a qualifier that flips which of two near-identical concepts is
right. It is far too slow to search a whole gazetteer with, so it runs only
over the shortlist the retrieval stage already produced.

Scope
-----
Adapted from the reference implementation's ``rerankers/cross_encoder.py``,
which is 777 lines. Only its ~45-line inference path is reproduced here.
Everything else in that file trains the model — BCE pairs, margin ranking,
knowledge-graph triplets, checkpoint saving — and this repository does no
training. The reference reaches its inference path by subclassing
``sentence_transformers.CrossEncoder`` through a training-oriented base class;
this wraps the same class instead, so none of that machinery comes along.

Score calibration
-----------------
``CrossEncoder.predict`` applies the model's own activation: a sigmoid for
single-logit rerankers, giving [0, 1], but identity for others, giving raw
logits on an unbounded scale. Since this score can become the CDM's
``concept_confidence``, ``normalise_scores`` maps whatever the model emits into
[0, 1], and ``RerankResult`` keeps the raw value for debugging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import device
from app.src.nel.schemas import MatchCandidate, MentionAnnotation

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 256

_RERANKER_CACHE: dict[Path, "CrossEncoderReranker"] = {}


@dataclass(frozen=True)
class RerankResult:
    """One reranked candidate and the raw model output behind it."""

    candidate: MatchCandidate
    raw_score: float


def _sigmoid(values: np.ndarray) -> np.ndarray:
    # Computed in the numerically stable halves so a large-magnitude logit
    # cannot overflow exp().
    out = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_negative = np.exp(values[~positive])
    out[~positive] = exp_negative / (1.0 + exp_negative)
    return out


def normalise_scores(scores: np.ndarray) -> np.ndarray:
    """Map cross-encoder output into [0, 1] without changing the ranking.

    Values already inside [0, 1] are a sigmoid output and pass through
    untouched. Anything outside is treated as a raw logit and squashed. Both
    transforms are monotonic, so reranking order is never affected — only the
    number reported as a confidence.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return scores
    if float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0:
        return scores
    return _sigmoid(scores)


class CrossEncoderReranker:
    """Rescores (mention, candidate term) pairs with a cross-encoder."""

    method = "cross_encoder"

    def __init__(
        self,
        model_path: str | Path,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        model: object | None = None,
    ) -> None:
        self.batch_size = batch_size
        if model is not None:
            # Injected for tests, so the suite needs no checkpoint.
            self.model = model
            return

        from sentence_transformers import CrossEncoder

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Cross-encoder not found at {model_path}. Register it under "
                "'rerank.<lang>' in the registry and run 'python -m app.model_manager'."
            )
        logger.info("Loading cross-encoder reranker: %s", model_path)
        self.model = CrossEncoder(str(model_path), max_length=max_length, device=device)

    def rerank(
        self,
        mentions: list[MentionAnnotation],
        candidates: list[list[MatchCandidate]],
        top_k: int | None = None,
    ) -> list[list[MatchCandidate]]:
        """Reorder each mention's candidates by cross-encoder score.

        All pairs across all mentions are scored in one batched call rather
        than one call per mention — the reason the shortlist is small enough
        for this to be affordable at all.

        Mentions with no candidates stay empty. Candidate objects are returned
        with ``score``, ``rank`` and ``method`` rewritten, and their prior
        values preserved under ``metadata``.
        """
        if len(mentions) != len(candidates):
            raise ValueError(
                f"Got {len(mentions)} mentions but {len(candidates)} candidate lists"
            )

        pairs: list[tuple[str, str]] = []
        spans: list[tuple[int, int]] = []
        for mention, mention_candidates in zip(mentions, candidates):
            start = len(pairs)
            pairs.extend((mention.text, candidate.term) for candidate in mention_candidates)
            spans.append((start, len(pairs)))

        if not pairs:
            return [[] for _ in mentions]

        raw = np.asarray(
            self.model.predict(
                pairs,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float64,
        ).reshape(-1)

        if raw.size != len(pairs):
            raise ValueError(
                f"Cross-encoder returned {raw.size} scores for {len(pairs)} pairs"
            )
        normalised = normalise_scores(raw)

        reranked: list[list[MatchCandidate]] = []
        for mention_candidates, (start, end) in zip(candidates, spans):
            if start == end:
                reranked.append([])
                continue

            order = np.argsort(normalised[start:end])[::-1]
            ordered: list[MatchCandidate] = []
            for rank, position in enumerate(order, 1):
                candidate = mention_candidates[int(position)]
                candidate.metadata = {
                    **candidate.metadata,
                    "pre_rerank_method": candidate.method,
                    "pre_rerank_score": candidate.score,
                    "pre_rerank_rank": candidate.rank,
                    "cross_encoder_raw": float(raw[start + int(position)]),
                }
                candidate.score = float(normalised[start + int(position)])
                candidate.method = self.method
                candidate.rank = rank
                ordered.append(candidate)
            reranked.append(ordered[:top_k] if top_k else ordered)
        return reranked


def get_reranker(model_path: str | Path, **kwargs) -> CrossEncoderReranker:
    """Load a reranker once per process and reuse it thereafter."""
    resolved = Path(model_path).resolve()
    cached = _RERANKER_CACHE.get(resolved)
    if cached is None:
        cached = CrossEncoderReranker(resolved, **kwargs)
        _RERANKER_CACHE[resolved] = cached
    return cached


def clear_cache() -> None:
    """Drop cached rerankers. Intended for tests."""
    _RERANKER_CACHE.clear()
