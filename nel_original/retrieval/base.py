"""Base interface for candidate retrieval methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import GazetteerEntry, MatchCandidate, MentionAnnotation


class BaseBiEncoder(ABC):
    """Base class for retrieval methods that build an index over a gazetteer."""

    def __init__(
        self,
        top_k: int = 25,
        threshold: float = 0.0,
        label_aware: bool = True,
    ) -> None:
        self.top_k = top_k
        self.threshold = threshold
        self.label_aware = label_aware
        self.gazetteer: list[GazetteerEntry] = []

    def build_index(self, gazetteer: list[GazetteerEntry]) -> None:
        """Store the gazetteer and build the method-specific index."""
        self.gazetteer = gazetteer
        self._build_index_internal(gazetteer)

    @abstractmethod
    def _build_index_internal(self, gazetteer: list[GazetteerEntry]) -> None:
        """Build the method-specific index."""

    @abstractmethod
    def search(
        self,
        mentions: list[MentionAnnotation],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[list[MatchCandidate]]:
        """Retrieve ranked candidates for each mention."""
