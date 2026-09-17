"""Typed records shared by matching, retrieval, reranking and evaluation."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MentionAnnotation:
    """One entity mention and its optional gold normalization code."""

    filename: str
    label: str | None
    start_span: int | None
    end_span: int | None
    text: str
    code: str | None
    is_abbreviation: bool | None = None
    is_composite: bool | None = None
    need_context: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GazetteerEntry:
    """One terminology or gazetteer entry."""

    term: str
    code: str
    label: str | None = None
    semantic_tag: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Concept:
    """Ontology concept with its preferred term and optional aliases."""

    code: str
    term: str
    label: str | None = None
    semantic_tag: str | None = None
    aliases: list[str] = field(default_factory=list)


@dataclass
class HierarchyEdge:
    """Directed parent-to-child hierarchy relation."""

    parent_code: str
    child_code: str
    relation: str | None = None


@dataclass
class MatchCandidate:
    """Candidate concept produced by a matcher, retriever or reranker."""

    mention_id: str | None
    filename: str | None
    text: str
    label: str | None
    code: str | None
    candidate_term: str | None
    score: float
    method: str
    rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LinkedEntity:
    """Final entity-linking output for one mention."""

    mention: MentionAnnotation
    candidates: list[MatchCandidate]
    predicted_code: str | None
    predicted_term: str | None
    score: float | None
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)
