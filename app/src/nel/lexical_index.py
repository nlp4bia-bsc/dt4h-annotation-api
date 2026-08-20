"""Persisted lexical indexes over a gazetteer: char n-gram TF-IDF and BM25.

These complement the dense retriever rather than replacing it.  A bi-encoder
generalises well but is weakest exactly where string overlap is decisive —
typos, inflection, word order, abbreviations, code-like forms.  A char n-gram
index catches ``"meningitis"`` from ``"meningitiss"``; embeddings may not.

Persistence
-----------
Unlike the FAISS index, the artifacts here are fitted scikit-learn estimators,
and pickled estimators carry no cross-version format guarantee — scikit-learn
says so explicitly and scipy's sparse containers have changed shape before.
Rather than hand-roll a version-neutral TF-IDF, each index records the
``scikit-learn`` and ``scipy`` versions that produced it.  On load, any
mismatch — or a changed gazetteer — triggers a rebuild instead of an unpickle.
A rebuild costs one pass over the gazetteer with no model involved, which is
cheap next to the risk of silently mis-loading a stale estimator.

Only files this application generated are ever unpickled.  They live under
``RESOURCES_PATH`` next to the FAISS indexes.

One artifact, two scorers
-------------------------
Both generators fit on the same rows in the same order as the FAISS index
(``gazetteer.load_gazetteer``), so a row position means the same concept
everywhere.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from app.src.nel.gazetteer import fingerprint, load_gazetteer
from app.src.nel.normalize import normalize_text

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1

# Char 3-5 grams: 3 is short enough to survive a one-character typo in a short
# word, 5 long enough to keep whole morphemes intact. max_features caps the
# vocabulary so a large gazetteer cannot grow an unbounded matrix.
CHAR_NGRAM_RANGE = (3, 5)
MAX_CHAR_FEATURES = 300_000
MAX_WORD_FEATURES = 200_000

# BM25 Okapi constants. k1 damps term-frequency saturation; b controls length
# normalisation.
BM25_K1 = 1.5
BM25_B = 0.75

_INDEX_CACHE: dict[Path, "LexicalIndex"] = {}


class LexicalIndexError(RuntimeError):
    """A lexical index could not be loaded or built."""


@dataclass(frozen=True)
class LexicalHit:
    """One scored gazetteer row."""

    row: int
    score: float


def _index_manifest_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".manifest.json")


def _normalise_terms(terms: list[str]) -> list[str]:
    """Lowercase and collapse whitespace before fitting or querying.

    Applied on both sides, so build and query always see the same form.  Unlike
    the dense path there is no trained model to disturb here — normalising is
    unambiguously right for string matching.
    """
    return [normalize_text(term, lowercase=True) for term in terms]


class LexicalIndex:
    """Char n-gram TF-IDF and word-level BM25 over one gazetteer.

    Built and persisted together because they share the row ordering and the
    same fit-time pass over the gazetteer; either scorer can be used alone.
    """

    def __init__(
        self,
        tfidf_vectorizer: TfidfVectorizer,
        tfidf_matrix: sp.csr_matrix,
        count_vectorizer: CountVectorizer,
        count_matrix: sp.csr_matrix,
        manifest: dict,
    ) -> None:
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_matrix = tfidf_matrix.tocsr()
        self.count_vectorizer = count_vectorizer
        self.manifest = manifest

        counts = count_matrix.tocsc()
        self._counts = counts
        n_docs = counts.shape[0]

        # Document frequency per token = number of rows with a non-zero count.
        doc_freq = np.diff(counts.indptr).astype(np.float64)
        # Okapi BM25 idf, in the +1 variant that stays positive for tokens
        # present in more than half the gazetteer.
        self._bm25_idf = np.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

        doc_len = np.asarray(count_matrix.sum(axis=1)).ravel().astype(np.float64)
        avg_len = float(doc_len.mean()) if n_docs else 0.0
        # Precompute the per-document denominator term, which does not depend
        # on the query, so scoring a mention is a few vector ops per token.
        self._bm25_norm = BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len / (avg_len or 1.0)))
        self._n_docs = n_docs

    @property
    def n_rows(self) -> int:
        """Number of gazetteer rows indexed."""
        return self.tfidf_matrix.shape[0]

    # -- scoring -------------------------------------------------------

    def tfidf_top_k(self, queries: list[str], k: int) -> list[list[LexicalHit]]:
        """Cosine similarity over char n-gram TF-IDF vectors.

        Both sides are L2-normalised by ``TfidfVectorizer``, so the dot product
        is the cosine directly.
        """
        if not queries:
            return []
        query_matrix = self.tfidf_vectorizer.transform(_normalise_terms(queries))
        similarities = (query_matrix @ self.tfidf_matrix.T).tocsr()

        results: list[list[LexicalHit]] = []
        for row_index in range(similarities.shape[0]):
            row = similarities.getrow(row_index)
            results.append(self._top_from_sparse_row(row, k))
        return results

    def bm25_top_k(self, queries: list[str], k: int) -> list[list[LexicalHit]]:
        """Okapi BM25 over word tokens.

        Note the shape of the data: gazetteer entries are terms, not documents,
        typically one to four words.  BM25's length normalisation was designed
        for documents and does comparatively little here, so this generally
        ranks below char n-gram TF-IDF for terminology matching.  It is
        provided because it costs almost nothing on top of the same matrix.
        """
        if not queries:
            return []

        results: list[list[LexicalHit]] = []
        for query in _normalise_terms(queries):
            scores = self._bm25_scores(query)
            results.append(self._top_from_dense(scores, k))
        return results

    def _bm25_scores(self, query: str) -> np.ndarray:
        analyzer = self.count_vectorizer.build_analyzer()
        vocabulary = self.count_vectorizer.vocabulary_

        columns = [vocabulary[token] for token in set(analyzer(query)) if token in vocabulary]
        scores = np.zeros(self._n_docs, dtype=np.float64)
        if not columns:
            return scores

        for column in columns:
            start, end = self._counts.indptr[column], self._counts.indptr[column + 1]
            rows = self._counts.indices[start:end]
            term_freq = self._counts.data[start:end].astype(np.float64)
            scores[rows] += self._bm25_idf[column] * (
                term_freq * (BM25_K1 + 1.0) / (term_freq + self._bm25_norm[rows])
            )
        return scores

    @staticmethod
    def _top_from_sparse_row(row: sp.csr_matrix, k: int) -> list[LexicalHit]:
        if row.nnz == 0:
            return []
        take = min(k, row.nnz)
        # argpartition finds the top `take` without sorting all nnz entries.
        positions = np.argpartition(row.data, -take)[-take:]
        positions = positions[np.argsort(row.data[positions])[::-1]]
        return [LexicalHit(row=int(row.indices[p]), score=float(row.data[p])) for p in positions]

    @staticmethod
    def _top_from_dense(scores: np.ndarray, k: int) -> list[LexicalHit]:
        nonzero = np.flatnonzero(scores)
        if nonzero.size == 0:
            return []
        take = min(k, nonzero.size)
        candidate_scores = scores[nonzero]
        positions = np.argpartition(candidate_scores, -take)[-take:]
        positions = positions[np.argsort(candidate_scores[positions])[::-1]]
        return [
            LexicalHit(row=int(nonzero[p]), score=float(candidate_scores[p])) for p in positions
        ]


# ----------------------------------------------------------------------
# Build / load
# ----------------------------------------------------------------------


def build(gaz_path: str | Path, index_path: str | Path) -> LexicalIndex:
    """Fit both vectorizers over a gazetteer and persist them."""
    gaz_path, index_path = Path(gaz_path), Path(index_path)
    rows = load_gazetteer(gaz_path)
    terms = _normalise_terms(rows["term"].tolist())

    logger.info("Building lexical index for %s (%d terms)", gaz_path.name, len(terms))

    tfidf_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=CHAR_NGRAM_RANGE,
        max_features=MAX_CHAR_FEATURES,
        dtype=np.float32,
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(terms)

    count_vectorizer = CountVectorizer(
        analyzer="word",
        max_features=MAX_WORD_FEATURES,
        dtype=np.int32,
    )
    count_matrix = count_vectorizer.fit_transform(terms)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as handle:
        pickle.dump(
            {
                "tfidf_vectorizer": tfidf_vectorizer,
                "tfidf_matrix": tfidf_matrix,
                "count_vectorizer": count_vectorizer,
                "count_matrix": count_matrix,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "n_rows": int(len(terms)),
        "char_ngram_range": list(CHAR_NGRAM_RANGE),
        "gazetteer_path": str(gaz_path),
        "gazetteer_sha256": fingerprint(gaz_path),
        "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    _index_manifest_path(index_path).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    logger.info(
        "Lexical index ready: %s (%d rows, %d char features, %d word features)",
        index_path.name, len(terms), tfidf_matrix.shape[1], count_matrix.shape[1],
    )
    return LexicalIndex(
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        count_vectorizer=count_vectorizer,
        count_matrix=count_matrix,
        manifest=manifest,
    )


def load_or_build(gaz_path: str | Path, index_path: str | Path) -> LexicalIndex:
    """Return a cached index, else load a valid one from disk, else build it.

    Any reason the persisted copy cannot be trusted — absent, no manifest,
    different gazetteer, different scikit-learn or scipy — results in a
    rebuild.  Nothing stale is ever returned.
    """
    index_path = Path(index_path).resolve()
    cached = _INDEX_CACHE.get(index_path)
    if cached is not None:
        return cached

    reason = _why_not_reusable(index_path, gaz_path)
    if reason is None:
        try:
            index = _load(index_path)
        except Exception as exc:  # noqa: BLE001 — any unpickle failure means rebuild
            logger.warning("Lexical index %s failed to load (%s) — rebuilding", index_path.name, exc)
            index = build(gaz_path, index_path)
    else:
        logger.info("Lexical index %s: %s — building", index_path.name, reason)
        index = build(gaz_path, index_path)

    _INDEX_CACHE[index_path] = index
    return index


def _why_not_reusable(index_path: Path, gaz_path: str | Path) -> str | None:
    """Return a human-readable reason to rebuild, or None to reuse."""
    if not index_path.exists():
        return "not built yet"

    manifest_path = _index_manifest_path(index_path)
    if not manifest_path.exists():
        return "no manifest"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "unreadable manifest"

    if manifest.get("manifest_version") != MANIFEST_VERSION:
        return f"manifest version {manifest.get('manifest_version')!r}"
    if manifest.get("sklearn_version") != sklearn.__version__:
        return f"built with scikit-learn {manifest.get('sklearn_version')}, running {sklearn.__version__}"
    if manifest.get("scipy_version") != scipy.__version__:
        return f"built with scipy {manifest.get('scipy_version')}, running {scipy.__version__}"
    if manifest.get("char_ngram_range") != list(CHAR_NGRAM_RANGE):
        return "char n-gram range changed"
    if manifest.get("gazetteer_sha256") != fingerprint(gaz_path):
        return "gazetteer has changed"
    return None


def _load(index_path: Path) -> LexicalIndex:
    manifest = json.loads(_index_manifest_path(index_path).read_text(encoding="utf-8"))
    # Only files this application wrote under RESOURCES_PATH reach here, and
    # the manifest check above has already confirmed the producing versions.
    with index_path.open("rb") as handle:
        payload = pickle.load(handle)

    index = LexicalIndex(
        tfidf_vectorizer=payload["tfidf_vectorizer"],
        tfidf_matrix=payload["tfidf_matrix"],
        count_vectorizer=payload["count_vectorizer"],
        count_matrix=payload["count_matrix"],
        manifest=manifest,
    )
    if index.n_rows != manifest.get("n_rows"):
        raise LexicalIndexError(
            f"{index_path.name}: holds {index.n_rows} rows, manifest says {manifest.get('n_rows')}"
        )
    logger.info("Loaded lexical index %s (%d rows)", index_path.name, index.n_rows)
    return index


def clear_cache() -> None:
    """Drop cached indexes. Intended for tests."""
    _INDEX_CACHE.clear()
