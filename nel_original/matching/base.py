"""Base interface for lexical entity matchers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

from ..preprocessing import normalize_text
from ..schemas import GazetteerEntry, MatchCandidate, MentionAnnotation


class BaseEntityMatcher(ABC):
    """Base class for matchers that score every compatible gazetteer entry."""

    method = "base"

    def __init__(
        self,
        gazetteer: list[GazetteerEntry],
        threshold: float = 0.0,
        top_k: int = 10,
        label_aware: bool = True,
        keep_duplicate_codes: bool = False,
        normalizer: Callable[[str], str] | None = None,
    ) -> None:
        self.gazetteer = gazetteer
        self.threshold = threshold
        self.top_k = top_k
        self.label_aware = label_aware
        self.keep_duplicate_codes = keep_duplicate_codes
        self.normalizer = normalizer or normalize_text

    def fit(self) -> "BaseEntityMatcher":
        """Build any method-specific index and return the matcher."""
        self._fit_internal()
        return self

    def _fit_internal(self) -> None:
        """Build an optional method-specific index."""

    def predict(self, mentions: list[MentionAnnotation]) -> list[list[MatchCandidate]]:
        """Score a batch of mentions."""
        return [self.score_mention(mention) for mention in mentions]

    def score_mention(self, mention: MentionAnnotation) -> list[MatchCandidate]:
        """Score one mention, deduplicate codes and assign one-based ranks."""
        candidates: list[MatchCandidate] = []
        for entry, score in self._score_candidates(mention):
            if score < self.threshold:
                continue
            candidates.append(
                MatchCandidate(
                    mention_id=None,
                    filename=mention.filename,
                    text=mention.text,
                    label=mention.label,
                    code=entry.code,
                    candidate_term=entry.term,
                    score=float(score),
                    method=self.method,
                )
            )

        if not self.keep_duplicate_codes:
            best_by_code: dict[str | None, MatchCandidate] = {}
            for candidate in candidates:
                current = best_by_code.get(candidate.code)
                if current is None or candidate.score > current.score:
                    best_by_code[candidate.code] = candidate
            candidates = list(best_by_code.values())

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        candidates = candidates[: self.top_k]
        for rank, candidate in enumerate(candidates, 1):
            candidate.rank = rank
        return candidates

    @abstractmethod
    def _score_candidates(
        self,
        mention: MentionAnnotation,
    ) -> Iterable[tuple[GazetteerEntry, float]]:
        """Yield gazetteer entries with their scores for one mention."""
