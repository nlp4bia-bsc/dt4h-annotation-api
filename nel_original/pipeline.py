"""Entity-linking pipeline orchestration."""

from __future__ import annotations

import inspect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .data import load_annotations_tsv, load_brat_dir
from .ensembling import reciprocal_rank_fusion
from .schemas import LinkedEntity, MatchCandidate, MentionAnnotation


class EntityLinkingPipeline:
    """Run one or more candidate generators followed by an optional reranker.

    Candidate generators may implement either ``predict(mentions)`` or
    ``search(mentions)``. Multiple generators are fused with RRF. A single
    generator preserves its original ranking and scores.
    """

    def __init__(
        self,
        candidate_generator: Any | None = None,
        *,
        candidate_generators: Sequence[Any] | None = None,
        reranker: Any | None = None,
        top_k_candidates: int = 25,
        top_k_final: int = 1,
        rrf_k: int = 60,
    ) -> None:
        generators = list(candidate_generators or [])
        if candidate_generator is not None:
            generators.insert(0, candidate_generator)
        if not generators:
            raise ValueError("At least one candidate generator is required")
        self.candidate_generators = generators
        self.candidate_generator = generators[0]
        self.reranker = reranker
        self.top_k_candidates = top_k_candidates
        self.top_k_final = top_k_final
        self.rrf_k = rrf_k

    def fit(self, gazetteer: Any | None = None) -> "EntityLinkingPipeline":
        """Fit or index each candidate generator when it exposes such a method."""
        for generator in self.candidate_generators:
            if hasattr(generator, "build_index") and gazetteer is not None:
                generator.build_index(gazetteer)
            elif hasattr(generator, "fit"):
                generator.fit()
        return self

    @staticmethod
    def _generate(generator: Any, mentions: list[MentionAnnotation]) -> list[list[MatchCandidate]]:
        if hasattr(generator, "predict"):
            return generator.predict(mentions)
        if hasattr(generator, "search"):
            return generator.search(mentions)
        raise TypeError(f"Candidate generator {type(generator).__name__} has neither predict nor search")

    def _rerank(self, mention: MentionAnnotation, candidates: list[MatchCandidate]) -> list[MatchCandidate]:
        if self.reranker is None or not candidates:
            return candidates

        rerank_parameters = inspect.signature(self.reranker.rerank).parameters
        if "k" not in rerank_parameters:
            result = self.reranker.rerank(mention, candidates)
            if not isinstance(result, list) or (result and not isinstance(result[0], MatchCandidate)):
                raise TypeError("The reranker must return a list of MatchCandidate objects")
            return result

        packed = [{"terms": [c.candidate_term or "" for c in candidates], "codes": [c.code or "" for c in candidates]}]
        result = self.reranker.rerank([mention.text], packed, k=self.top_k_candidates)[0]
        score_by_code = {
            str(code): float(score) for code, score in zip(result.get("codes", []), result.get("similarity", []))
        }
        by_code = {str(candidate.code): candidate for candidate in candidates if candidate.code is not None}
        reranked = []
        for rank, code in enumerate(result.get("codes", []), 1):
            candidate = by_code[str(code)]
            candidate.score = score_by_code[str(code)]
            candidate.method = "cross_encoder"
            candidate.rank = rank
            reranked.append(candidate)
        return reranked

    def link_mentions(self, mentions: list[MentionAnnotation]) -> list[LinkedEntity]:
        """Link a list of mentions and return candidates plus the top prediction."""
        generated = [self._generate(generator, mentions) for generator in self.candidate_generators]
        output: list[LinkedEntity] = []
        for mention_index, mention in enumerate(mentions):
            if len(generated) == 1:
                candidates = list(generated[0][mention_index])[: self.top_k_candidates]
            else:
                source_rankings = {
                    getattr(generator, "method", generator.__class__.__name__): generated[index][mention_index]
                    for index, generator in enumerate(self.candidate_generators)
                }
                candidates = reciprocal_rank_fusion(source_rankings, k=self.rrf_k, top_k=self.top_k_candidates)
            candidates = self._rerank(mention, candidates)[: self.top_k_candidates]
            top = candidates[0] if candidates else None
            output.append(
                LinkedEntity(
                    mention=mention,
                    candidates=candidates,
                    predicted_code=top.code if top else None,
                    predicted_term=top.candidate_term if top else None,
                    score=top.score if top else None,
                    method=top.method if top else "none",
                )
            )
        return output

    def link_from_tsv(self, path: str | Path) -> list[LinkedEntity]:
        """Load mention annotations from TSV and link them."""
        return self.link_mentions(load_annotations_tsv(str(path)))

    def link_from_brat_dir(self, path: str | Path) -> list[LinkedEntity]:
        """Load BRAT annotations from a directory and link them."""
        return self.link_mentions(load_brat_dir(str(path)))

    def save_outputs(self, linked: Sequence[LinkedEntity], path: str | Path) -> Path:
        """Write one row per mention with the final prediction and all candidates."""
        rows = []
        for entity in linked:
            top = entity.candidates[0] if entity.candidates else None
            rows.append(
                {
                    "filename": entity.mention.filename,
                    "start_span": entity.mention.start_span,
                    "end_span": entity.mention.end_span,
                    "text": entity.mention.text,
                    "gold_code": entity.mention.code,
                    "predicted_code": top.code if top else None,
                    "predicted_term": top.candidate_term if top else None,
                    "score": top.score if top else None,
                    "method": top.method if top else "none",
                    "candidates_json": json.dumps(
                        [candidate.__dict__ for candidate in entity.candidates], ensure_ascii=False
                    ),
                }
            )
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)
        return output_path
