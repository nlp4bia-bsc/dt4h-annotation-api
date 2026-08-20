from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import log
from typing import Callable, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .base import BaseEntityMatcher

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler, Levenshtein
except ImportError:  # pragma: no cover
    fuzz = Levenshtein = JaroWinkler = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover
    TfidfVectorizer = None


# ---------------------------------------------------------------------------
# Existing BaseEntityMatcher API
# ---------------------------------------------------------------------------


class StringMatchMatcher(BaseEntityMatcher):
    """Score candidates by normalized exact string equality."""

    method = "string_match"

    def _score_candidates(self, mention):
        mention_text = self.normalizer(mention.text)

        for entity in self.gazetteer:
            if self.label_aware and mention.label and entity.label and mention.label != entity.label:
                continue

            yield entity, float(self.normalizer(entity.term) == mention_text)


class LevenshteinMatcher(BaseEntityMatcher):
    """Score candidates with normalized Levenshtein similarity."""

    method = "levenshtein"

    def _score_candidates(self, mention):
        mention_text = self.normalizer(mention.text)

        for entity in self.gazetteer:
            if self.label_aware and mention.label and entity.label and mention.label != entity.label:
                continue

            entity_text = self.normalizer(entity.term)
            score = (
                Levenshtein.normalized_similarity(mention_text, entity_text)
                if Levenshtein is not None
                else SequenceMatcher(None, mention_text, entity_text).ratio()
            )
            yield entity, float(score)


class JaroWinklerMatcher(BaseEntityMatcher):
    """Score candidates with Jaro-Winkler similarity."""

    method = "jaro_winkler"

    def _score_candidates(self, mention):
        mention_text = self.normalizer(mention.text)

        for entity in self.gazetteer:
            if self.label_aware and mention.label and entity.label and mention.label != entity.label:
                continue

            entity_text = self.normalizer(entity.term)
            score = (
                JaroWinkler.similarity(mention_text, entity_text)
                if JaroWinkler is not None
                else SequenceMatcher(None, mention_text, entity_text).ratio()
            )
            if score > 1.0:
                score /= 100.0
            yield entity, float(score)


class TokenSetMatcher(BaseEntityMatcher):
    """Score candidates using token-set similarity."""

    method = "token_set"

    def _score_candidates(self, mention):
        mention_text = self.normalizer(mention.text)

        for entity in self.gazetteer:
            if self.label_aware and mention.label and entity.label and mention.label != entity.label:
                continue

            entity_text = self.normalizer(entity.term)
            score = (
                fuzz.token_set_ratio(mention_text, entity_text) / 100.0
                if fuzz is not None
                else SequenceMatcher(
                    None,
                    " ".join(sorted(mention_text.split())),
                    " ".join(sorted(entity_text.split())),
                ).ratio()
            )
            yield entity, float(score)


class TfidfCharNgramMatcher(BaseEntityMatcher):
    """TF-IDF char n-gram version for the existing BaseEntityMatcher API."""

    method = "tfidf_char_ngram"

    def __init__(
        self,
        *args,
        ngram_range: tuple[int, int] = (3, 5),
        analyzer: str = "char_wb",
        **kwargs,
    ) -> None:
        if TfidfVectorizer is None:
            raise ImportError("TfidfCharNgramMatcher requires scikit-learn. Install it with: pip install scikit-learn")
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self._tfidf_ready = False
        super().__init__(*args, **kwargs)

    def _build_index(self) -> None:
        self._entities = list(self.gazetteer)
        terms = [self.normalizer(entity.term) or "" for entity in self._entities]
        self._vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            lowercase=False,
            norm="l2",
            dtype=np.float32,
        )
        self._matrix = self._vectorizer.fit_transform(terms)
        self._tfidf_ready = True

    def _score_candidates(self, mention):
        if not self._tfidf_ready:
            self._build_index()

        query = self._vectorizer.transform([self.normalizer(mention.text) or ""])
        scores = (self._matrix @ query.T).toarray().ravel()

        for entity, score in zip(self._entities, scores):
            if self.label_aware and mention.label and entity.label and mention.label != entity.label:
                continue
            yield entity, float(score)


class BM25Matcher(BaseEntityMatcher):
    """Classic BM25 version for the existing BaseEntityMatcher API."""

    method = "bm25"

    def __init__(
        self,
        *args,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], list[str]]] = None,
        **kwargs,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero.")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1.")

        self.k1 = float(k1)
        self.b = float(b)
        self.tokenizer = tokenizer or (lambda value: value.split())
        self._bm25_ready = False
        super().__init__(*args, **kwargs)

    def _build_index(self) -> None:
        self._entities = list(self.gazetteer)
        tokenized_terms = [self.tokenizer(self.normalizer(entity.term) or "") for entity in self._entities]
        self._doc_lengths = [len(tokens) for tokens in tokenized_terms]
        self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0

        document_frequency: Counter[str] = Counter()
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for entity_idx, tokens in enumerate(tokenized_terms):
            frequencies = Counter(tokens)
            document_frequency.update(frequencies.keys())
            for token, frequency in frequencies.items():
                self._postings[token].append((entity_idx, frequency))

        n_docs = len(tokenized_terms)
        self._idf = {
            token: log(1.0 + (n_docs - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        self._bm25_ready = True

    def _score_candidates(self, mention):
        if not self._bm25_ready:
            self._build_index()

        scores = [0.0] * len(self._entities)
        if self._avg_doc_length <= 0.0:
            for entity, score in zip(self._entities, scores):
                yield entity, score
            return

        for token in set(self.tokenizer(self.normalizer(mention.text) or "")):
            token_idf = self._idf.get(token)
            if token_idf is None:
                continue

            for entity_idx, term_frequency in self._postings[token]:
                doc_length = self._doc_lengths[entity_idx]
                denominator = term_frequency + self.k1 * (1.0 - self.b + self.b * doc_length / self._avg_doc_length)
                scores[entity_idx] += token_idf * term_frequency * (self.k1 + 1.0) / denominator

        for entity, score in zip(self._entities, scores):
            if self.label_aware and mention.label and entity.label and mention.label != entity.label:
                continue
            yield entity, float(score)


WhooshContextMatcher = LevenshteinMatcher


# ---------------------------------------------------------------------------
# Dataframe retrieval API: test/gold rows -> codes, candidates, scores
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"\w+(?:[-_/]\w+)*", flags=re.UNICODE)
TERM_COLUMN_CANDIDATES = ("term", "text", "span", "candidate", "descriptor")


def default_normalizer(text: object) -> str:
    """Fallback normalizer when the project-level normalizer is not provided."""
    text = str(text).lower().strip()
    return re.sub(r"\s+", " ", text)


def _safe_string(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def _resolve_term_column(
    dataframe: pd.DataFrame,
    requested_column: Optional[str],
    dataframe_name: str,
) -> str:
    if requested_column is not None:
        if requested_column not in dataframe.columns:
            raise ValueError(
                f"{dataframe_name} does not contain {requested_column!r}. Available columns: {list(dataframe.columns)}"
            )
        return requested_column

    for column in TERM_COLUMN_CANDIDATES:
        if column in dataframe.columns:
            return column

    raise ValueError(
        f"Could not identify a term column in {dataframe_name}. "
        f"Expected one of {TERM_COLUMN_CANDIDATES}; "
        f"available: {list(dataframe.columns)}"
    )


def combine_gazetteer_and_train(
    gazetteer_df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    *,
    gazetteer_term_col: Optional[str] = None,
    train_term_col: Optional[str] = None,
    code_col: str = "code",
    label_col: str = "label",
    normalizer: Callable[[object], str] = default_normalizer,
) -> pd.DataFrame:
    """Combine a gazetteer and train aliases into one candidate catalogue.

    The returned dataframe has one row per term/code/label alias combination:

        term, code, label, source, norm_term

    Duplicate aliases are removed. If the same normalised alias maps to the
    same code in both sources, the gazetteer copy is retained.
    """

    def prepare(
        dataframe: pd.DataFrame,
        *,
        source: str,
        term_col: Optional[str],
        dataframe_name: str,
    ) -> pd.DataFrame:
        """Normalize one candidate source to the shared catalogue schema."""
        resolved_term_col = _resolve_term_column(
            dataframe,
            term_col,
            dataframe_name,
        )
        if code_col not in dataframe.columns:
            raise ValueError(f"{dataframe_name} must contain {code_col!r}.")

        required = [resolved_term_col, code_col]
        if label_col in dataframe.columns:
            required.append(label_col)

        output = dataframe.loc[:, required].copy()
        output = output.rename(columns={resolved_term_col: "term", code_col: "code"})
        if label_col in output.columns:
            output = output.rename(columns={label_col: "label"})
        else:
            output["label"] = None

        output = output.dropna(subset=["term", "code"])
        output["term"] = output["term"].astype(str).str.strip()
        output["code"] = output["code"].astype(str).str.strip()
        output["label"] = output["label"].map(_safe_string)
        output = output[(output["term"] != "") & (output["code"] != "")].copy()
        output["norm_term"] = output["term"].map(normalizer)
        output = output[output["norm_term"] != ""].copy()
        output["source"] = source
        output["source_priority"] = 0 if source == "gazetteer" else 1
        return output

    parts = [
        prepare(
            gazetteer_df,
            source="gazetteer",
            term_col=gazetteer_term_col,
            dataframe_name="gazetteer_df",
        )
    ]
    if train_df is not None:
        parts.append(
            prepare(
                train_df,
                source="train",
                term_col=train_term_col,
                dataframe_name="train_df",
            )
        )

    catalogue = pd.concat(parts, ignore_index=True)
    catalogue = catalogue.sort_values(
        ["source_priority", "term", "code"],
        kind="stable",
    )
    catalogue = catalogue.drop_duplicates(
        subset=["norm_term", "code", "label"],
        keep="first",
    ).reset_index(drop=True)

    return catalogue[["term", "code", "label", "source", "norm_term"]]


@dataclass(frozen=True)
class CandidateRecord:
    """One dataframe-oriented lexical candidate."""

    code: str
    candidate: str
    score: float


class BaseDataFrameLexicalRetriever:
    """Base class for dataframe lexical retrieval without evaluation logic."""

    method = "base"

    def __init__(
        self,
        gazetteer_df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
        *,
        gazetteer_term_col: Optional[str] = None,
        train_term_col: Optional[str] = None,
        code_col: str = "code",
        label_col: str = "label",
        normalizer: Callable[[object], str] = default_normalizer,
        label_aware: bool = True,
    ) -> None:
        self.code_col = code_col
        self.label_col = label_col
        self.normalizer = normalizer
        self.label_aware = label_aware

        self.catalogue = combine_gazetteer_and_train(
            gazetteer_df=gazetteer_df,
            train_df=train_df,
            gazetteer_term_col=gazetteer_term_col,
            train_term_col=train_term_col,
            code_col=code_col,
            label_col=label_col,
            normalizer=normalizer,
        )
        if self.catalogue.empty:
            raise ValueError("The combined gazetteer/train catalogue is empty.")

        self._terms = self.catalogue["term"].to_numpy(dtype=object)
        self._norm_terms = self.catalogue["norm_term"].to_numpy(dtype=object)
        self._codes = self.catalogue["code"].to_numpy(dtype=object)
        self._labels = self.catalogue["label"].to_numpy(dtype=object)
        self._build_index()

    def _build_index(self) -> None:
        raise NotImplementedError

    def _rank_one(
        self,
        normalized_query: str,
        query_label: Optional[str],
        top_k: int,
        min_score: float,
    ) -> list[CandidateRecord]:
        raise NotImplementedError

    def predict(
        self,
        query_df: pd.DataFrame,
        *,
        text_col: str = "text",
        top_k: int = 200,
        min_score: float = 0.0,
        show_progress: bool = True,
        progress_desc: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return input rows with ranked ``candidates``, ``codes`` and ``scores``.

        ``query_df`` may be a gold dataframe, a dev dataframe or a test
        dataframe. Its rows and existing columns remain unchanged.
        """
        if text_col not in query_df.columns:
            raise ValueError(f"query_df must contain {text_col!r}. Available columns: {list(query_df.columns)}")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        output = query_df.copy()
        codes_out: list[list[str]] = []
        candidates_out: list[list[str]] = []
        scores_out: list[list[float]] = []
        has_label = self.label_col in output.columns

        # ``iterrows`` is deliberate here: it supports any valid dataframe
        # column name, unlike attribute access through ``itertuples``.
        rows = output.iterrows()
        if show_progress:
            rows = tqdm(
                rows,
                total=len(output),
                desc=progress_desc or f"{self.method}: queries",
                unit="mention",
                dynamic_ncols=True,
                leave=True,
            )

        for _, row in rows:
            query_label = _safe_string(row[self.label_col]) if has_label else None
            ranking = self._rank_one(
                normalized_query=self.normalizer(row[text_col]),
                query_label=query_label,
                top_k=top_k,
                min_score=min_score,
            )
            codes_out.append([item.code for item in ranking])
            candidates_out.append([item.candidate for item in ranking])
            scores_out.append([float(item.score) for item in ranking])

        output["candidates"] = candidates_out
        output["codes"] = codes_out
        output["scores"] = scores_out
        return output

    def _is_allowed(self, alias_idx: int, query_label: Optional[str]) -> bool:
        if not self.label_aware or query_label is None:
            return True

        candidate_label = _safe_string(self._labels[alias_idx])
        return candidate_label is None or candidate_label == query_label

    def _deduplicate_by_code(
        self,
        alias_indices: np.ndarray,
        alias_scores: np.ndarray,
        query_label: Optional[str],
        top_k: int,
        min_score: float,
    ) -> list[CandidateRecord]:
        """Keep the highest-scoring candidate alias for every unique code."""
        best_by_code: dict[str, tuple[float, str]] = {}

        for alias_idx, score in zip(alias_indices, alias_scores):
            alias_idx = int(alias_idx)
            score = float(score)
            if score < min_score or not self._is_allowed(alias_idx, query_label):
                continue

            code = str(self._codes[alias_idx])
            candidate = str(self._terms[alias_idx])
            current = best_by_code.get(code)

            if current is None or score > current[0] or (score == current[0] and candidate < current[1]):
                best_by_code[code] = (score, candidate)

        ranking = [
            CandidateRecord(code=code, candidate=candidate, score=score)
            for code, (score, candidate) in best_by_code.items()
        ]
        ranking.sort(key=lambda item: (-item.score, item.code, item.candidate))
        return ranking[:top_k]


class TfidfCharNgramRetriever(BaseDataFrameLexicalRetriever):
    """TF-IDF character n-gram retrieval; scores are cosine similarities."""

    method = "tfidf_char_ngram"

    def __init__(
        self,
        *args,
        ngram_range: tuple[int, int] = (3, 5),
        analyzer: str = "char_wb",
        **kwargs,
    ) -> None:
        if TfidfVectorizer is None:
            raise ImportError(
                "TfidfCharNgramRetriever requires scikit-learn. Install it with: pip install scikit-learn"
            )
        if analyzer not in {"char", "char_wb"}:
            raise ValueError("analyzer must be 'char' or 'char_wb'.")
        if ngram_range[0] <= 0 or ngram_range[0] > ngram_range[1]:
            raise ValueError("ngram_range must be a valid positive (min_n, max_n).")

        self.ngram_range = ngram_range
        self.analyzer = analyzer
        super().__init__(*args, **kwargs)

    def _build_index(self) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            lowercase=False,
            norm="l2",
            dtype=np.float32,
        )
        self.term_matrix = self.vectorizer.fit_transform(self._norm_terms)

    def _rank_one(
        self,
        normalized_query: str,
        query_label: Optional[str],
        top_k: int,
        min_score: float,
    ) -> list[CandidateRecord]:
        if not normalized_query:
            return []

        query_vector = self.vectorizer.transform([normalized_query])
        # Only aliases with a non-zero character n-gram overlap are returned.
        similarities = (query_vector @ self.term_matrix.T).tocsr()
        return self._deduplicate_by_code(
            alias_indices=similarities.indices,
            alias_scores=similarities.data,
            query_label=query_label,
            top_k=top_k,
            min_score=min_score,
        )


class BM25Retriever(BaseDataFrameLexicalRetriever):
    """Classic token-based BM25 retrieval with a reusable inverted index."""

    method = "bm25"

    def __init__(
        self,
        *args,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], list[str]]] = None,
        **kwargs,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero.")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1.")

        self.k1 = float(k1)
        self.b = float(b)
        self.tokenizer = tokenizer or TOKEN_RE.findall
        super().__init__(*args, **kwargs)

    def _build_index(self) -> None:
        tokenized_terms = [self.tokenizer(term) for term in self._norm_terms]
        self.doc_lengths = np.asarray(
            [len(tokens) for tokens in tokenized_terms],
            dtype=np.float32,
        )
        self.avg_doc_length = float(self.doc_lengths.mean()) if len(self.doc_lengths) else 0.0

        document_frequency: Counter[str] = Counter()
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for alias_idx, tokens in enumerate(tokenized_terms):
            frequencies = Counter(tokens)
            document_frequency.update(frequencies.keys())
            for token, frequency in frequencies.items():
                self.postings[token].append((alias_idx, frequency))

        n_docs = len(tokenized_terms)
        self.idf = {
            token: log(1.0 + (n_docs - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def _rank_one(
        self,
        normalized_query: str,
        query_label: Optional[str],
        top_k: int,
        min_score: float,
    ) -> list[CandidateRecord]:
        if not normalized_query or self.avg_doc_length <= 0.0:
            return []

        query_tokens = set(self.tokenizer(normalized_query))
        if not query_tokens:
            return []

        scores_by_alias: dict[int, float] = defaultdict(float)
        for token in query_tokens:
            token_idf = self.idf.get(token)
            if token_idf is None:
                continue

            for alias_idx, term_frequency in self.postings[token]:
                doc_length = float(self.doc_lengths[alias_idx])
                denominator = term_frequency + self.k1 * (1.0 - self.b + self.b * doc_length / self.avg_doc_length)
                scores_by_alias[alias_idx] += token_idf * term_frequency * (self.k1 + 1.0) / denominator

        if not scores_by_alias:
            return []

        alias_indices = np.fromiter(scores_by_alias.keys(), dtype=np.int64)
        alias_scores = np.fromiter(scores_by_alias.values(), dtype=np.float32)
        return self._deduplicate_by_code(
            alias_indices=alias_indices,
            alias_scores=alias_scores,
            query_label=query_label,
            top_k=top_k,
            min_score=min_score,
        )
