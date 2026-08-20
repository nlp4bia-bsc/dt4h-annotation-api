"""Public API for nlp4bia-linking."""

from .pipeline import EntityLinkingPipeline
from .retrieval import CandidateRetrievalPipeline, RetrievalResult, build_vocabulary
from .schemas import Concept, GazetteerEntry, HierarchyEdge, LinkedEntity, MatchCandidate, MentionAnnotation

__version__ = "0.2.0"

__all__ = [
    "CandidateRetrievalPipeline",
    "Concept",
    "EntityLinkingPipeline",
    "GazetteerEntry",
    "HierarchyEdge",
    "LinkedEntity",
    "MatchCandidate",
    "MentionAnnotation",
    "RetrievalResult",
    "build_vocabulary",
    "__version__",
]
