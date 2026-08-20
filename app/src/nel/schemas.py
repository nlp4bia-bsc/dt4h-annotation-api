"""Typed records shared by the NEL candidate generators and the fuser.

The pipeline's public contract is still ``list[list[dict]]`` — these types are
internal to linking.  They exist so that more than one candidate generator can
be combined: fusion needs a rank and a source per candidate, and bare parallel
lists of codes/terms/scores cannot carry that.

Ported from the reference implementation's ``schemas.py``, trimmed to the three
records the API actually uses.  ``LinkedEntity``, ``Concept`` and
``HierarchyEdge`` were left behind — the first duplicates the annotation dict we
already emit, the other two exist only for ontology evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MentionAnnotation:
    """One entity mention awaiting normalisation.

    ``text`` is the surface form as the NER stage found it.  ``label`` is the
    NER class, kept so a label-aware generator can refuse cross-class matches.
    """

    text: str
    label: str | None = None
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True)
class GazetteerEntry:
    """One terminology entry: a surface form and the concept it denotes."""

    term: str
    code: str
    label: str | None = None


@dataclass
class MatchCandidate:
    """A candidate concept produced by one generator for one mention.

    Mutable by design: fusion rewrites ``score``, ``rank`` and ``method`` while
    merging rankings, and a future reranker overwrites the same fields.

    ``rank`` is 1-based and is what reciprocal-rank fusion consumes; ``score``
    is the generator's own similarity and is *not* comparable across
    generators, which is exactly why fusion works on ranks instead.
    """

    code: str
    term: str
    score: float
    method: str
    rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
