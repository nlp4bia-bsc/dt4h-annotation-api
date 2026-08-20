from __future__ import annotations

import ast
import json
import random
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Iterable, Literal, Optional

import networkx as nx
import numpy as np
import pandas as pd

from ..schemas import Concept

Direction = Literal["ascending", "descending", "both"]


def parse_list_column(value: Any) -> list[Any]:
    """Convert a cell into a list.

    Supports real Python lists/tuples, JSON strings, Python-literal strings and
    scalar values. Empty/NA values become an empty list.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if pd.isna(value):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            return json.loads(value)
        except Exception:
            pass
        try:
            return ast.literal_eval(value)
        except Exception:
            pass
    return [value]


def normalize_code(code: Any) -> str:
    """Normalize ontology identifiers as strings."""
    return str(code)


def normalized_name(value: Any) -> str:
    """Normalize names for duplicate/same-name safeguards."""
    return " ".join(str(value).casefold().split())


def deduplicate_preserve_order(items: Iterable[Any]) -> list[Any]:
    """Deduplicate hashable items preserving first occurrence order."""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def canonical_pair_key(left: str, right: str) -> tuple[str, str]:
    """Return an order-invariant normalized key for sentence-similarity pairs."""
    return tuple(sorted((normalized_name(left), normalized_name(right))))


def dedupe_code_term_pairs(codes: Iterable[Any], candidates: Iterable[Any]) -> list[tuple[str, str]]:
    """Deduplicate candidate ``(term, code)`` pairs by code preserving rank."""
    pairs: list[tuple[str, str]] = []
    seen_codes: set[str] = set()
    for code, term in zip(codes, candidates):
        code = normalize_code(code)
        if code in seen_codes:
            continue
        seen_codes.add(code)
        pairs.append((str(term), code))
    return pairs


def build_graph_adjacency(
    graph: nx.DiGraph | None, edge_direction: str = "parent_to_child"
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build child→parents and parent→children maps from a directed ontology graph."""
    child_to_parents: dict[str, set[str]] = {}
    parent_to_children: dict[str, set[str]] = {}
    if graph is None:
        return child_to_parents, parent_to_children
    for left, right in graph.edges():
        parent, child = (left, right) if edge_direction == "parent_to_child" else (right, left)
        parent = normalize_code(parent)
        child = normalize_code(child)
        child_to_parents.setdefault(child, set()).add(parent)
        parent_to_children.setdefault(parent, set()).add(child)
    return child_to_parents, parent_to_children


def graph_related_codes_by_distance(
    code: str,
    max_depth: int,
    mode: Direction,
    child_to_parents: dict[str, set[str]],
    parent_to_children: dict[str, set[str]],
) -> dict[int, set[str]]:
    """Return related codes grouped by graph distance for compatibility callers."""
    if mode == "ascending":
        maps = (child_to_parents,)
    elif mode == "descending":
        maps = (parent_to_children,)
    elif mode == "both":
        maps = (child_to_parents, parent_to_children)
    else:
        raise ValueError("mode must be 'ascending', 'descending' or 'both'")
    current = {normalize_code(code)}
    visited = {normalize_code(code)}
    out: dict[int, set[str]] = {}
    for depth in range(1, max_depth + 1):
        nxt: set[str] = set()
        for node in current:
            for mapping in maps:
                nxt.update(mapping.get(node, set()))
        nxt -= visited
        if not nxt:
            current = set()
            continue
        out[depth] = nxt
        visited.update(nxt)
        current = nxt
    return out


class TripletGenerationStrategy(ABC):
    """Base class for anchor-positive-negative triplet generation."""

    def __init__(
        self,
        df: pd.DataFrame,
        anchor_col: str = "term",
        code_col: str = "code",
        codes_col: str = "codes",
        candidates_col: str = "candidates",
        scores_col: str = "scores",
    ):
        self.df = df.copy()
        self.anchor_col = anchor_col
        self.code_col = code_col
        self.codes_col = codes_col
        self.candidates_col = candidates_col
        self.scores_col = scores_col

    @abstractmethod
    def generate(self) -> pd.DataFrame:
        """Generate triplets for the configured dataframe."""
        raise NotImplementedError

    def _make_triplet_row(
        self,
        anchor: str,
        positive: str,
        negative: str,
        strategy: str,
        positive_code: Optional[str] = None,
        negative_code: Optional[str] = None,
    ) -> dict[str, Any]:
        return {
            "anchor": anchor,
            "positive": positive,
            "negative": negative,
            "strategy": strategy,
            "positive_code": positive_code,
            "negative_code": negative_code,
        }


class RandomTripletGenerator(TripletGenerationStrategy):
    """Generate random negatives efficiently from a gazetteer dataframe."""

    def __init__(
        self,
        df: pd.DataFrame,
        gazetteer_df: pd.DataFrame,
        num_negatives: int = 199,
        random_state: Optional[int] = 42,
        gazetteer_code_col: str = "code",
        gazetteer_term_col: str = "term",
        **kwargs: Any,
    ):
        super().__init__(df, **kwargs)
        self.num_negatives = int(num_negatives)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        gaz = gazetteer_df[[gazetteer_code_col, gazetteer_term_col]].dropna().drop_duplicates().copy()
        gaz[gazetteer_code_col] = gaz[gazetteer_code_col].astype(str)
        gaz[gazetteer_term_col] = gaz[gazetteer_term_col].astype(str)
        self.gaz_codes = gaz[gazetteer_code_col].to_numpy()
        self.gaz_terms = gaz[gazetteer_term_col].to_numpy()
        self.n_gaz = len(gaz)
        if self.n_gaz == 0:
            raise ValueError("gazetteer_df is empty after removing nulls and duplicates.")
        self.code_to_first_term = gaz.groupby(gazetteer_code_col)[gazetteer_term_col].first().to_dict()

    def _sample_negative_indices(self, positive_code: str) -> np.ndarray:
        """Sample gazetteer indices whose code differs from the positive code."""
        selected: list[int] = []
        while len(selected) < self.num_negatives:
            remaining = self.num_negatives - len(selected)
            sample_size = min(self.n_gaz, max(remaining * 3, remaining + 50))
            idx = self.rng.choice(self.n_gaz, size=sample_size, replace=False)
            idx = idx[self.gaz_codes[idx] != positive_code]
            selected.extend(idx.tolist())
            selected = deduplicate_preserve_order(selected)
            if len(selected) < self.num_negatives and sample_size == self.n_gaz:
                break
        if len(selected) < self.num_negatives:
            raise ValueError(
                f"Not enough negatives for positive_code={positive_code}. "
                f"requested={self.num_negatives}, available={len(selected)}"
            )
        return np.array(selected[: self.num_negatives], dtype=int)

    def generate(self) -> pd.DataFrame:
        """Generate random-negative triplets."""
        rows: list[dict[str, Any]] = []
        for row in self.df.itertuples(index=False):
            anchor = str(getattr(row, self.anchor_col))
            positive_code = normalize_code(getattr(row, self.code_col))
            positive_text = self.code_to_first_term.get(positive_code, anchor)
            neg_idx = self._sample_negative_indices(positive_code)
            for i in neg_idx:
                rows.append(
                    self._make_triplet_row(
                        anchor=anchor,
                        positive=str(positive_text),
                        negative=str(self.gaz_terms[i]),
                        strategy="random",
                        positive_code=positive_code,
                        negative_code=str(self.gaz_codes[i]),
                    )
                )
        return pd.DataFrame(rows)


class SimilarityTripletGenerator(TripletGenerationStrategy):
    """Generate triplets from previously retrieved candidates."""

    def __init__(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
        max_negatives: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(df, **kwargs)
        self.threshold = threshold
        self.max_negatives = max_negatives

    def _extract_row_items(self, row: Any) -> tuple[str, str, list[str], list[str], list[Optional[float]]]:
        anchor = str(getattr(row, self.anchor_col))
        positive_code = normalize_code(getattr(row, self.code_col))
        codes = [normalize_code(c) for c in parse_list_column(getattr(row, self.codes_col))]
        candidates = [str(c) for c in parse_list_column(getattr(row, self.candidates_col))]
        if hasattr(row, self.scores_col):
            scores = parse_list_column(getattr(row, self.scores_col))
            scores = [None if s is None else float(s) for s in scores]
        else:
            scores = [None] * len(codes)
        if not (len(codes) == len(candidates) == len(scores)):
            raise ValueError(
                f"Longitudes incompatibles en fila anchor={anchor!r}: "
                f"codes={len(codes)}, candidates={len(candidates)}, scores={len(scores)}"
            )
        return anchor, positive_code, codes, candidates, scores

    def _get_positive_index(self, positive_code: str, codes: list[str]) -> Optional[int]:
        try:
            return codes.index(positive_code)
        except ValueError:
            return None

    def _get_similarity_negatives(
        self,
        positive_code: str,
        codes: list[str],
        candidates: list[str],
        scores: list[Optional[float]],
    ) -> list[tuple[str, str, Optional[float]]]:
        negatives: list[tuple[str, str, Optional[float]]] = []
        for code, candidate, score in zip(codes, candidates, scores):
            if code == positive_code:
                continue
            if self.threshold is not None and score is not None and score <= self.threshold:
                continue
            negatives.append((code, candidate, score))
        negatives = deduplicate_preserve_order(negatives)
        if self.max_negatives is not None:
            negatives = negatives[: self.max_negatives]
        return negatives

    def generate(self) -> pd.DataFrame:
        """Generate similarity-mined triplets."""
        rows: list[dict[str, Any]] = []
        for row in self.df.itertuples(index=False):
            anchor, positive_code, codes, candidates, scores = self._extract_row_items(row)
            positive_index = self._get_positive_index(positive_code, codes)
            if positive_index is None:
                continue
            positive_text = candidates[positive_index]
            negatives = self._get_similarity_negatives(positive_code, codes, candidates, scores)
            for negative_code, negative_text, _score in negatives:
                rows.append(
                    self._make_triplet_row(
                        anchor=anchor,
                        positive=positive_text,
                        negative=negative_text,
                        strategy="similarity",
                        positive_code=positive_code,
                        negative_code=negative_code,
                    )
                )
        return pd.DataFrame(rows)


class HierarchyTripletGenerator(SimilarityTripletGenerator):
    """Enrich similarity negatives with ontology neighbours explored by BFS."""

    def __init__(
        self,
        df: pd.DataFrame,
        graph: nx.DiGraph,
        gazetteer_df: pd.DataFrame,
        threshold: Optional[float] = None,
        direction: Direction = "ascending",
        depth: int = 1,
        max_negatives: Optional[int] = None,
        gazetteer_code_col: str = "code",
        gazetteer_term_col: str = "term",
        include_original_similarity_negatives: bool = True,
        **kwargs: Any,
    ):
        super().__init__(df=df, threshold=threshold, max_negatives=max_negatives, **kwargs)
        if not isinstance(graph, nx.DiGraph):
            raise TypeError("graph must be a networkx.DiGraph")
        if direction not in {"ascending", "descending", "both"}:
            raise ValueError("direction must be 'ascending', 'descending' or 'both'")
        if depth < 0:
            raise ValueError("depth must be >= 0")
        self.graph = graph
        self.gazetteer_df = gazetteer_df.copy()
        self.direction = direction
        self.depth = depth
        self.gazetteer_code_col = gazetteer_code_col
        self.gazetteer_term_col = gazetteer_term_col
        self.include_original_similarity_negatives = include_original_similarity_negatives
        self.gazetteer_df[self.gazetteer_code_col] = self.gazetteer_df[self.gazetteer_code_col].astype(str)
        self.gazetteer_df[self.gazetteer_term_col] = self.gazetteer_df[self.gazetteer_term_col].astype(str)
        self.code_to_terms = (
            self.gazetteer_df.groupby(self.gazetteer_code_col)[self.gazetteer_term_col]
            .apply(lambda x: deduplicate_preserve_order([str(v) for v in x]))
            .to_dict()
        )

    def _neighbors(self, code: str) -> Iterable[str]:
        """Return predecessors/successors according to the configured direction."""
        if code not in self.graph:
            return []
        if self.direction == "ascending":
            return self.graph.predecessors(code)
        if self.direction == "descending":
            return self.graph.successors(code)
        return list(self.graph.predecessors(code)) + list(self.graph.successors(code))

    def _expand_code(self, start_code: str, positive_code: str) -> list[str]:
        """Expand a code by BFS, including the start code and excluding positive_code."""
        start_code = normalize_code(start_code)
        positive_code = normalize_code(positive_code)
        if start_code == positive_code:
            return []
        visited = {start_code}
        expanded = [start_code]
        queue = deque([(start_code, 0)])
        while queue:
            current_code, current_depth = queue.popleft()
            if current_depth >= self.depth:
                continue
            for neighbor in self._neighbors(current_code):
                neighbor = normalize_code(neighbor)
                if neighbor == positive_code or neighbor in visited:
                    continue
                visited.add(neighbor)
                expanded.append(neighbor)
                queue.append((neighbor, current_depth + 1))
        return expanded

    def _codes_to_terms(
        self,
        codes: Iterable[str],
        fallback_terms_by_code: Optional[dict[str, str]] = None,
    ) -> list[tuple[str, str]]:
        """Convert codes to ``(code, term)`` using gazetteer terms or fallbacks."""
        fallback_terms_by_code = fallback_terms_by_code or {}
        out: list[tuple[str, str]] = []
        for code in codes:
            code = normalize_code(code)
            terms = self.code_to_terms.get(code)
            if terms:
                for term in terms:
                    out.append((code, term))
            elif code in fallback_terms_by_code:
                out.append((code, fallback_terms_by_code[code]))
        return deduplicate_preserve_order(out)

    def generate(self) -> pd.DataFrame:
        """Generate hierarchy-enriched similarity triplets."""
        rows: list[dict[str, Any]] = []
        for row in self.df.itertuples(index=False):
            anchor, positive_code, codes, candidates, scores = self._extract_row_items(row)
            positive_index = self._get_positive_index(positive_code, codes)
            if positive_index is None:
                continue
            positive_text = candidates[positive_index]
            similarity_negatives = self._get_similarity_negatives(positive_code, codes, candidates, scores)
            fallback_terms_by_code = {
                code: candidate for code, candidate, _score in similarity_negatives if code != positive_code
            }
            enriched_negative_codes: list[str] = []
            for negative_code, _negative_text, _score in similarity_negatives:
                expanded_codes = self._expand_code(start_code=negative_code, positive_code=positive_code)
                enriched_negative_codes.extend(expanded_codes)
            enriched_negative_codes = deduplicate_preserve_order(enriched_negative_codes)
            enriched_negative_pairs = self._codes_to_terms(
                enriched_negative_codes, fallback_terms_by_code=fallback_terms_by_code
            )
            if not self.include_original_similarity_negatives:
                original_negative_codes = {code for code, _text, _score in similarity_negatives}
                enriched_negative_pairs = [
                    (code, term) for code, term in enriched_negative_pairs if code not in original_negative_codes
                ]
            for negative_code, negative_text in enriched_negative_pairs:
                if negative_code == positive_code or negative_text == positive_text:
                    continue
                rows.append(
                    self._make_triplet_row(
                        anchor=anchor,
                        positive=positive_text,
                        negative=negative_text,
                        strategy=f"hierarchy_{self.direction}_depth_{self.depth}",
                        positive_code=positive_code,
                        negative_code=negative_code,
                    )
                )
        return pd.DataFrame(rows)


# Compatibility layer used by older scripts/notebooks/tests.
def selected_positive_from_retrieval(code: str, retrieval: Iterable[tuple[str, str]]) -> Optional[str]:
    """Return the first retrieved term whose code matches the anchor code."""
    for term, cand_code in retrieval:
        if normalize_code(cand_code) == normalize_code(code):
            return str(term)
    return None


def clean_negative_candidates(
    negatives: Iterable[tuple[str, str, str]],
    anchor: str,
    positive: str,
    anchor_code: str,
    limit: int | None = None,
) -> list[tuple[str, str, str]]:
    """Filter invalid and duplicate negative candidates while preserving order."""
    positive_names = {normalized_name(anchor), normalized_name(positive)}
    clean: list[tuple[str, str, str]] = []
    seen_codes: set[str] = set()
    seen_names: set[str] = set()
    for neg_term, neg_code, source in negatives:
        neg_code = normalize_code(neg_code)
        name = normalized_name(neg_term)
        if (
            neg_code == normalize_code(anchor_code)
            or not name
            or name in positive_names
            or name in seen_names
            or neg_code in seen_codes
        ):
            continue
        clean.append((str(neg_term), neg_code, str(source)))
        seen_codes.add(neg_code)
        seen_names.add(name)
        if limit is not None and len(clean) >= limit:
            break
    return clean


class RetrievalCandidateRecord:
    """Candidate list for one annotated mention."""

    def __init__(self, mention: str, gold_code: str, retrieval: list[tuple[str, str]]):
        self.mention = mention
        self.gold_code = gold_code
        self.retrieval = retrieval


def build_cross_encoder_training_rows(
    records: list[RetrievalCandidateRecord | dict[str, Any]],
    code_to_terms: dict[str, list[str]],
    strategy: str,
    retriever_method: str,
    max_negatives_per_anchor: int = 199,
    graph_expansion_mode: Direction = "ascending",
    child_to_parents: dict[str, set[str]] | None = None,
    parent_to_children: dict[str, set[str]] | None = None,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compatibility wrapper around the strategy classes.

    New code should instantiate ``RandomTripletGenerator``,
    ``SimilarityTripletGenerator`` or ``HierarchyTripletGenerator`` directly.
    """
    rows = []
    for record in records:
        mention = record.mention if isinstance(record, RetrievalCandidateRecord) else str(record["mention"])
        gold_code = record.gold_code if isinstance(record, RetrievalCandidateRecord) else str(record["gold_code"])
        retrieval = record.retrieval if isinstance(record, RetrievalCandidateRecord) else list(record["retrieval"])
        rows.append(
            {
                "term": mention,
                "code": gold_code,
                "codes": [code for _term, code in retrieval],
                "candidates": [term for term, _code in retrieval],
                "scores": [None] * len(retrieval),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        empty_pairs = pd.DataFrame(
            columns=[
                "strategy",
                "retriever_method",
                "mention",
                "candidate",
                "label",
                "mention_code",
                "candidate_code",
                "source",
            ]
        )
        empty_triplets = pd.DataFrame(columns=["anchor", "positive", "negative"])
        return (
            empty_pairs,
            empty_triplets,
            {
                "graph_expansion_mode": graph_expansion_mode,
                "anchors_with_positive": 0,
                "skipped_without_positive": 0,
                "mean_negatives_per_anchor": 0.0,
                "max_negatives_per_anchor": 0,
            },
        )

    if strategy == "random":
        gaz_rows = [(code, term) for code, terms in code_to_terms.items() for term in terms]
        generator = RandomTripletGenerator(
            df=df,
            gazetteer_df=pd.DataFrame(gaz_rows, columns=["code", "term"]),
            num_negatives=max_negatives_per_anchor,
            random_state=seed,
        )
    elif strategy == "similarity":
        generator = SimilarityTripletGenerator(df=df, threshold=None, max_negatives=max_negatives_per_anchor)
    elif strategy in {"parent-similarity", "grandparent-parent-similarity"}:
        depth = 1 if strategy == "parent-similarity" else 2
        graph = nx.DiGraph()
        child_to_parents = child_to_parents or {}
        parent_to_children = parent_to_children or {}
        if graph_expansion_mode in {"ascending", "both"}:
            for child, parents in child_to_parents.items():
                for parent in parents:
                    graph.add_edge(parent, child)
        if graph_expansion_mode in {"descending", "both"}:
            for parent, children in parent_to_children.items():
                for child in children:
                    graph.add_edge(parent, child)
        gaz_rows = [(code, term) for code, terms in code_to_terms.items() for term in terms]
        generator = HierarchyTripletGenerator(
            df=df,
            graph=graph,
            gazetteer_df=pd.DataFrame(gaz_rows, columns=["code", "term"]),
            threshold=None,
            direction=graph_expansion_mode,
            depth=depth,
            max_negatives=max_negatives_per_anchor,
        )
    else:
        raise ValueError(f"Unknown strategy={strategy!r}")

    triplets = generator.generate().drop_duplicates(["anchor", "positive", "negative"])
    # Keep the parquet triplets strict but derive BCE pair metadata for training summaries.
    pos = (
        triplets[["anchor", "positive", "positive_code"]]
        .drop_duplicates()
        .rename(columns={"anchor": "mention", "positive": "candidate", "positive_code": "candidate_code"})
    )
    pos["label"] = 1
    pos["mention_code"] = pos["candidate_code"]
    pos["source"] = "positive_from_retrieval"
    neg = (
        triplets[["anchor", "negative", "negative_code", "positive_code"]]
        .drop_duplicates()
        .rename(
            columns={
                "anchor": "mention",
                "negative": "candidate",
                "negative_code": "candidate_code",
                "positive_code": "mention_code",
            }
        )
    )
    neg["label"] = 0
    neg["source"] = strategy
    pairs = pd.concat([pos, neg], ignore_index=True)
    pairs["strategy"] = strategy
    pairs["retriever_method"] = retriever_method
    pairs = pairs[
        ["strategy", "retriever_method", "mention", "candidate", "label", "mention_code", "candidate_code", "source"]
    ]
    anchors_with_positive = 0
    if "codes" in df:
        anchors_with_positive = int(
            df.apply(
                lambda row: normalize_code(row["code"]) in [normalize_code(code) for code in row["codes"]], axis=1
            ).sum()
        )
    stats = {
        "graph_expansion_mode": graph_expansion_mode,
        "anchors_with_positive": anchors_with_positive,
        "skipped_without_positive": int(len(df) - anchors_with_positive),
        "mean_negatives_per_anchor": float(triplets.groupby("anchor").size().mean()) if not triplets.empty else 0.0,
        "max_negatives_per_anchor": int(triplets.groupby("anchor").size().max()) if not triplets.empty else 0,
    }
    return pairs, triplets[["anchor", "positive", "negative"]], stats


class TripletGenerator:
    """Generic concept triplet generator kept for CLI/backwards compatibility."""

    def __init__(
        self,
        concepts: list[Concept],
        seed: int = 13,
        negatives_per_positive: int = 1,
        max_negatives_per_concept: int = 199,
        add_positive_samples: bool = False,
        max_positive_sample_terms: int = 4,
        max_negative_sample_terms: int = 4,
    ):
        self.concepts = concepts
        self.r = random.Random(seed)
        self.negatives_per_positive = min(negatives_per_positive, max_negatives_per_concept)
        self.max_negatives_per_concept = max_negatives_per_concept
        self.add_positive_samples = add_positive_samples
        self.max_positive_sample_terms = max_positive_sample_terms
        self.max_negative_sample_terms = max_negative_sample_terms

    def _code_to_terms(self) -> dict[str, list[str]]:
        by: dict[str, list[str]] = {}
        for concept in self.concepts:
            by.setdefault(str(concept.code), [])
            for term in [concept.term, *concept.aliases]:
                term = str(term)
                if term and term not in by[str(concept.code)]:
                    by[str(concept.code)].append(term)
        return by

    def _add_triplet(
        self,
        rows: list[tuple[str, str, str]],
        seen: set[tuple[tuple[str, str], tuple[str, str]]],
        anchor: str,
        positive: str,
        negative: str,
    ) -> bool:
        if normalized_name(negative) in {normalized_name(anchor), normalized_name(positive)}:
            return False
        key = (canonical_pair_key(anchor, positive), canonical_pair_key(anchor, negative))
        if key in seen:
            return False
        seen.add(key)
        rows.append((anchor, positive, negative))
        return True

    def generate(self):
        """Generate legacy triplet rows from the selected strategy."""
        by = self._code_to_terms()
        positives = []
        seen_positive_pairs = set()
        for code, terms in by.items():
            if not terms:
                continue
            canonical_positive = terms[0]
            for term in terms:
                key = (code, canonical_pair_key(term, canonical_positive))
                if key in seen_positive_pairs:
                    continue
                seen_positive_pairs.add(key)
                positives.append((term, canonical_positive, code, "same_code"))
        negatives = []
        triplets = []
        seen_negative_pairs = set()
        seen_triplets = set()
        all_candidates = [concept for concept in self.concepts if str(concept.code) in by]
        for anchor, positive, anchor_code, _source in positives:
            pool = [concept for concept in all_candidates if str(concept.code) != str(anchor_code)]
            if not pool:
                continue
            self.r.shuffle(pool)
            added_for_anchor = 0
            for negative in pool:
                if added_for_anchor >= self.negatives_per_positive:
                    break
                negative_code = str(negative.code)
                negative_terms = by.get(negative_code, [str(negative.term)])
                negative_term = str(negative_terms[0])
                anchor_positive_names = {normalized_name(t) for t in by[str(anchor_code)]}
                anchor_positive_names.update({normalized_name(anchor), normalized_name(positive)})
                if normalized_name(negative_term) in anchor_positive_names:
                    continue
                neg_pair_key = (str(anchor_code), negative_code, canonical_pair_key(anchor, negative_term))
                if neg_pair_key in seen_negative_pairs:
                    continue
                seen_negative_pairs.add(neg_pair_key)
                negatives.append((anchor, negative_term, anchor_code, negative_code, "random"))
                added_for_anchor += 1
                positive_terms = (
                    by[str(anchor_code)][: self.max_positive_sample_terms] if self.add_positive_samples else [positive]
                )
                sampled_negative_terms = (
                    negative_terms[: self.max_negative_sample_terms] if self.add_positive_samples else [negative_term]
                )
                for positive_term in positive_terms:
                    for sampled_negative in sampled_negative_terms:
                        self._add_triplet(triplets, seen_triplets, anchor, positive_term, sampled_negative)
        return positives, negatives, triplets


def export_tsv(rows, cols, path):
    """Write generated rows to a TSV file with the requested columns."""
    pd.DataFrame(rows, columns=cols).drop_duplicates().to_csv(path, sep="\t", index=False)
