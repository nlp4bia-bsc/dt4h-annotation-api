"""Lexical entity matching methods."""

from .base import BaseEntityMatcher
from .lexical import (
    BM25Matcher,
    BM25Retriever,
    JaroWinklerMatcher,
    LevenshteinMatcher,
    StringMatchMatcher,
    TfidfCharNgramMatcher,
    TfidfCharNgramRetriever,
    TokenSetMatcher,
    WhooshContextMatcher,
)
from .registry import MATCHER_REGISTRY, build_matcher

__all__ = [
    "BaseEntityMatcher",
    "BM25Matcher",
    "BM25Retriever",
    "JaroWinklerMatcher",
    "LevenshteinMatcher",
    "MATCHER_REGISTRY",
    "StringMatchMatcher",
    "TfidfCharNgramMatcher",
    "TfidfCharNgramRetriever",
    "TokenSetMatcher",
    "WhooshContextMatcher",
    "build_matcher",
]
