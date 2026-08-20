"""Small registry for the built-in lexical matchers."""

from __future__ import annotations

from typing import Any

from .lexical import (
    BM25Matcher,
    JaroWinklerMatcher,
    LevenshteinMatcher,
    StringMatchMatcher,
    TfidfCharNgramMatcher,
    TokenSetMatcher,
    WhooshContextMatcher,
)

MATCHER_REGISTRY = {
    "string_match": StringMatchMatcher,
    "levenshtein": LevenshteinMatcher,
    "jaro_winkler": JaroWinklerMatcher,
    "token_set": TokenSetMatcher,
    "tfidf_char": TfidfCharNgramMatcher,
    "bm25": BM25Matcher,
    "whoosh": WhooshContextMatcher,
}


def build_matcher(method: str, **kwargs: Any) -> Any:
    """Instantiate a built-in matcher by its stable method name."""
    try:
        matcher_class = MATCHER_REGISTRY[method]
    except KeyError as exc:
        available = ", ".join(sorted(MATCHER_REGISTRY))
        raise ValueError(f"Unknown matcher '{method}'. Available methods: {available}") from exc
    return matcher_class(**kwargs)
