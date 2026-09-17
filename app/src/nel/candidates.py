"""Candidate generators: mention → ranked concept candidates.

Every generator implements one method::

    generate(mentions: list[MentionAnnotation], k: int) -> list[list[MatchCandidate]]

One inner list per mention, ordered best-first, ``rank`` filled in 1-based.
Returning fewer than *k* is normal and not an error.

The single explicit method is a deliberate departure from the reference
implementation, which duck-typed ``predict`` / ``search`` / ``get_candidates``
across three incompatible families.  That is why its best retriever could not
be driven by its own orchestrator: ``EntityLinkingPipeline._generate`` probes
for ``predict`` then ``search`` and raises on anything else, while
``HerbertFaissBiEncoder`` exposes only ``get_candidates`` — and returns bare
parallel lists rather than candidate objects.  One name, one return type.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from app.src.nel import lexical_index, vector_store
from app.src.nel.gazetteer import load_gazetteer
from app.src.nel.normalize import normalize_text
from app.src.nel.schemas import MatchCandidate, MentionAnnotation

logger = logging.getLogger(__name__)


class CandidateGenerator(Protocol):
    """Produces ranked concept candidates for a batch of mentions."""

    method: str

    def generate(
        self, mentions: list[MentionAnnotation], k: int
    ) -> list[list[MatchCandidate]]:
        """Return up to *k* candidates for each mention, best first."""
        ...


class DenseGenerator:
    """Dense retrieval over the persisted FAISS index.

    Mentions are passed to the encoder verbatim — see ``normalize.py`` for why
    the dense path does not normalise.
    """

    method = "biencoder"

    def __init__(self, gaz_path: Path, model_path: Path, index_path: Path) -> None:
        self.encoder = vector_store.get_encoder(model_path)
        self.store = vector_store.load(
            index_path=index_path, gaz_path=gaz_path, model_path=model_path
        )

    def generate(
        self, mentions: list[MentionAnnotation], k: int
    ) -> list[list[MatchCandidate]]:
        if not mentions:
            return []

        hits_per_mention = self.store.search(
            [mention.text for mention in mentions], encoder=self.encoder, k=k
        )
        return [
            [
                MatchCandidate(
                    code=hit.code,
                    term=hit.term,
                    score=hit.score,
                    method=self.method,
                    rank=rank,
                )
                for rank, hit in enumerate(hits, 1)
            ]
            for hits in hits_per_mention
        ]


class ExactMatchGenerator:
    """Exact lookup of the normalised surface form against the gazetteer.

    High precision, no recall beyond what is literally written in the
    gazetteer, and no index to build or persist — the lookup table is a dict
    over the same rows the FAISS index was built from.

    It earns its place next to the dense retriever because embedding similarity
    is not guaranteed to rank a verbatim match first: short mentions,
    abbreviations and code-like forms are exactly where a bi-encoder is
    weakest, and exactly where string equality is decisive.

    Emits at most one candidate per mention, always with ``score=1.0``.  That
    score is not comparable with a cosine similarity, which is precisely why
    fusion combines ranks rather than scores.
    """

    method = "exact_match"

    def __init__(self, gaz_path: Path, *, strip_accents: bool = False) -> None:
        self.strip_accents = strip_accents
        rows = load_gazetteer(gaz_path)

        self._lookup: dict[str, tuple[str, str]] = {}
        for term, code in zip(rows["term"], rows["code"]):
            key = self._key(term)
            # First row wins, matching load_gazetteer's own dedup rule so that
            # both paths resolve an ambiguous term to the same code.
            self._lookup.setdefault(key, (code, term))

        logger.debug(
            "ExactMatchGenerator: %d keys from %d gazetteer rows (%s)",
            len(self._lookup), len(rows), Path(gaz_path).name,
        )

    def _key(self, text: str) -> str:
        return normalize_text(text, lowercase=True, strip_accents=self.strip_accents)

    def generate(
        self, mentions: list[MentionAnnotation], k: int
    ) -> list[list[MatchCandidate]]:
        results: list[list[MatchCandidate]] = []
        for mention in mentions:
            found = self._lookup.get(self._key(mention.text))
            if found is None:
                results.append([])
                continue
            code, term = found
            results.append(
                [MatchCandidate(code=code, term=term, score=1.0, method=self.method, rank=1)]
            )
        return results


class _LexicalGenerator:
    """Shared plumbing for the sparse lexical generators.

    Both scorers read the same persisted index and the same gazetteer rows, so
    a returned row position resolves to a concept the same way in either.
    """

    method = "lexical"

    def __init__(self, gaz_path: Path, index_path: Path) -> None:
        self.index = lexical_index.load_or_build(gaz_path=gaz_path, index_path=index_path)
        rows = load_gazetteer(gaz_path)
        self._codes: list[str] = rows["code"].tolist()
        self._terms: list[str] = rows["term"].tolist()

        if len(self._codes) != self.index.n_rows:
            raise ValueError(
                f"Gazetteer resolves to {len(self._codes)} rows but the lexical index "
                f"holds {self.index.n_rows} — delete {Path(index_path).name} and retry."
            )

    def _score(self, queries: list[str], k: int):
        raise NotImplementedError

    def generate(
        self, mentions: list[MentionAnnotation], k: int
    ) -> list[list[MatchCandidate]]:
        if not mentions:
            return []

        # Overfetch: many gazetteer rows share a code, so the top rows can
        # collapse to far fewer than k distinct concepts.
        hits_per_mention = self._score([mention.text for mention in mentions], k * 5)

        results: list[list[MatchCandidate]] = []
        for hits in hits_per_mention:
            candidates: list[MatchCandidate] = []
            seen: set[str] = set()
            for hit in hits:
                code = self._codes[hit.row]
                if code in seen:
                    continue
                seen.add(code)
                candidates.append(
                    MatchCandidate(
                        code=code,
                        term=self._terms[hit.row],
                        score=hit.score,
                        method=self.method,
                        rank=len(candidates) + 1,
                    )
                )
                if len(candidates) >= k:
                    break
            results.append(candidates)
        return results


class TfidfCharNgramGenerator(_LexicalGenerator):
    """Char n-gram TF-IDF cosine — tolerant of typos, inflection and word order."""

    method = "tfidf_char"

    def _score(self, queries: list[str], k: int):
        return self.index.tfidf_top_k(queries, k)


class BM25Generator(_LexicalGenerator):
    """Okapi BM25 over word tokens.

    Weaker than ``TfidfCharNgramGenerator`` for terminology: BM25's length
    normalisation targets documents, and gazetteer entries are one to four
    words. Offered for completeness; prefer char n-grams unless measurement
    says otherwise.
    """

    method = "bm25"

    def _score(self, queries: list[str], k: int):
        return self.index.bm25_top_k(queries, k)
