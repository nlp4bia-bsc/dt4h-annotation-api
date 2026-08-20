from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..preprocessing import normalize_text
from ..schemas import GazetteerEntry, MatchCandidate, MentionAnnotation
from .base import BaseBiEncoder


class MatrixBiEncoder(BaseBiEncoder):
    """Sparse TF-IDF bi-encoder with efficient top-k candidate extraction.

    The original implementation materialized the full query × gazetteer score
    matrix and sorted every candidate for every mention. That is simple but slow
    for large ontologies. This version keeps the sparse matrix representation,
    extracts only the highest-scoring non-zero candidates with ``argpartition``
    and optionally ensembles char n-gram and word n-gram TF-IDF spaces.
    When `threshold <= 0`, zero-score fallback candidates are appended so the
    returned list reaches the requested number of unique codes whenever the
    gazetteer contains enough unique codes.
    """

    def __init__(
        self,
        similarity: str = "cosine",
        char_ngram_range: tuple[int, int] = (3, 5),
        word_ngram_range: tuple[int, int] = (1, 2),
        use_word_ngrams: bool = True,
        char_weight: float = 1.0,
        word_weight: float = 0.25,
        max_char_features: int | None = None,
        max_word_features: int | None = 200_000,
        strip_accents: bool = False,
        normalize_punct: bool = False,
        overfetch_factor: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.similarity = similarity
        self.char_ngram_range = char_ngram_range
        self.word_ngram_range = word_ngram_range
        self.use_word_ngrams = use_word_ngrams
        self.char_weight = char_weight
        self.word_weight = word_weight
        self.max_char_features = max_char_features
        self.max_word_features = max_word_features
        self.strip_accents = strip_accents
        self.normalize_punct = normalize_punct
        self.overfetch_factor = max(1, overfetch_factor)
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.char_ngram_range,
            max_features=self.max_char_features,
            dtype=np.float32,
        )
        self.word_vectorizer = (
            TfidfVectorizer(
                analyzer="word",
                ngram_range=self.word_ngram_range,
                max_features=self.max_word_features,
                dtype=np.float32,
            )
            if self.use_word_ngrams
            else None
        )
        self.term_matrix = None
        self.word_term_matrix = None

    def _normalize(self, text: str) -> str:
        return normalize_text(
            text,
            strip_accents=self.strip_accents,
            normalize_punct=self.normalize_punct,
        )

    def _build_index_internal(self, gazetteer: list[GazetteerEntry]) -> None:
        terms = [self._normalize(g.term) for g in gazetteer]
        self.term_matrix = self.vectorizer.fit_transform(terms)
        if self.word_vectorizer is not None:
            self.word_term_matrix = self.word_vectorizer.fit_transform(terms)

    def _score_queries(self, mentions: list[MentionAnnotation]):
        if self.term_matrix is None:
            raise RuntimeError("Index not built")
        texts = [self._normalize(m.text) for m in mentions]
        char_q = self.vectorizer.transform(texts)
        sims = (char_q @ self.term_matrix.T) * self.char_weight
        if self.word_vectorizer is not None and self.word_term_matrix is not None and self.word_weight:
            word_q = self.word_vectorizer.transform(texts)
            sims = sims + (word_q @ self.word_term_matrix.T) * self.word_weight
        return sims.tocsr()

    def _top_indices_from_sparse_row(self, row: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
        if row.nnz == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        fetch = min(row.nnz, max(k * self.overfetch_factor, k))
        data = row.data
        indices = row.indices
        if fetch < row.nnz:
            top_pos = np.argpartition(data, -fetch)[-fetch:]
            top_pos = top_pos[np.argsort(data[top_pos])[::-1]]
        else:
            top_pos = np.argsort(data)[::-1]
        return indices[top_pos], data[top_pos]

    def _make_candidate(
        self, mention: MentionAnnotation, entry: GazetteerEntry, score: float, rank: int
    ) -> MatchCandidate:
        return MatchCandidate(
            None,
            mention.filename,
            mention.text,
            mention.label,
            entry.code,
            entry.term,
            score,
            "matrix_biencoder",
            rank=rank,
            metadata={
                "candidate_span": entry.term,
                "char_weight": self.char_weight,
                "word_weight": self.word_weight if self.word_vectorizer is not None else 0.0,
            },
        )

    def _unique_code_rank(
        self,
        mention: MentionAnnotation,
        candidate_indices: np.ndarray,
        candidate_scores: np.ndarray,
        k: int,
        threshold: float,
        fill_zero: bool = True,
    ) -> list[MatchCandidate]:
        out: list[MatchCandidate] = []
        seen: set[str] = set()
        for idx, score in zip(candidate_indices, candidate_scores):
            g = self.gazetteer[int(idx)]
            if self.label_aware and mention.label and g.label and mention.label != g.label:
                continue
            value = float(score)
            if value < threshold:
                continue
            if g.code in seen:
                continue
            seen.add(g.code)
            out.append(self._make_candidate(mention, g, value, len(out) + 1))
            if len(out) >= k:
                break
        if fill_zero and len(out) < k and threshold <= 0.0:
            self._fill_zero_score_unique_codes(mention, out, seen, k)
        return out

    def _fill_zero_score_unique_codes(
        self,
        mention: MentionAnnotation,
        out: list[MatchCandidate],
        seen: set[str],
        k: int,
    ) -> None:
        """Fill sparse zero-score gaps so top-k means top-k unique codes when possible."""
        for entry in self.gazetteer:
            if len(out) >= k:
                break
            if self.label_aware and mention.label and entry.label and mention.label != entry.label:
                continue
            if entry.code in seen:
                continue
            seen.add(entry.code)
            out.append(self._make_candidate(mention, entry, 0.0, len(out) + 1))

    def search(
        self,
        mentions: list[MentionAnnotation],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[list[MatchCandidate]]:
        """Retrieve unique ranked codes from the sparse TF-IDF index."""
        k = top_k or self.top_k
        thr = self.threshold if threshold is None else threshold
        sims = self._score_queries(mentions)
        results: list[list[MatchCandidate]] = []
        for row_id, mention in enumerate(mentions):
            row = sims.getrow(row_id)
            idx, scores = self._top_indices_from_sparse_row(row, k)
            ranked = self._unique_code_rank(mention, idx, scores, k, thr, fill_zero=False)
            if len(ranked) < k and len(idx) < row.nnz:
                all_idx, all_scores = self._top_indices_from_sparse_row(row, row.nnz)
                ranked = self._unique_code_rank(mention, all_idx, all_scores, k, thr, fill_zero=True)
            elif len(ranked) < k and thr <= 0.0:
                seen = {candidate.code for candidate in ranked if candidate.code is not None}
                self._fill_zero_score_unique_codes(mention, ranked, seen, k)
            results.append(ranked)
        return results
