from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

from ..preprocessing import normalize_text
from ..schemas import GazetteerEntry, MatchCandidate, MentionAnnotation
from ..utils.device import optional_module, resolve_faiss_device
from .matrix import MatrixBiEncoder


class FaissBiEncoder(MatrixBiEncoder):
    """FAISS bi-encoder index with GPU-first execution and memory-safe vectorization.

    FAISS indexes require dense vectors. A plain TF-IDF char n-gram matrix can easily
    explode in memory for biomedical gazetteers (hundreds of thousands of rows ×
    hundreds of thousands of n-grams). For that reason the FAISS backend defaults to
    `HashingVectorizer` with a bounded number of features, avoiding vocabulary growth
    and preventing accidental 100+ GiB dense conversions.
    """

    def __init__(
        self,
        similarity: str = "cosine",
        device: str = "auto",
        vectorizer_backend: str = "hashing",
        n_features: int = 4096,
        max_features: int = 50000,
        analyzer: str = "char",
        ngram_range: tuple[int, int] = (3, 5),
        **kwargs: Any,
    ) -> None:
        super().__init__(similarity=similarity, **kwargs)
        self.device = device
        self.resolved_device = "cpu"
        self.index: Any | None = None
        self._term_vectors: np.ndarray | None = None
        self.vectorizer_backend = vectorizer_backend
        self.n_features = n_features
        self.max_features = max_features
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.vectorizer = self._build_vectorizer()

    def _build_vectorizer(self) -> Any:
        if self.vectorizer_backend == "hashing":
            return HashingVectorizer(
                analyzer=self.analyzer,
                ngram_range=self.ngram_range,
                n_features=self.n_features,
                alternate_sign=False,
                norm=None,
                dtype=np.float32,
            )
        if self.vectorizer_backend == "tfidf":
            return TfidfVectorizer(
                analyzer=self.analyzer,
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                dtype=np.float32,
            )
        raise ValueError("vectorizer_backend must be 'hashing' or 'tfidf'")

    def _require_faiss(self) -> Any:
        faiss = optional_module("faiss")
        if faiss is None:
            raise ImportError("faiss-gpu (preferred) or faiss-cpu is required for FaissBiEncoder")
        return faiss

    def _build_index_internal(self, gazetteer: list[GazetteerEntry]) -> None:
        faiss = self._require_faiss()
        terms = [normalize_text(g.term) for g in gazetteer]
        sparse_matrix = self.vectorizer.fit_transform(terms)
        vectors = sparse_matrix.astype(np.float32).toarray()
        if self.similarity == "cosine":
            faiss.normalize_L2(vectors)
        dim = vectors.shape[1]
        cpu_index = faiss.IndexFlatIP(dim) if self.similarity in {"cosine", "dot"} else faiss.IndexFlatL2(dim)
        cpu_index.add(vectors)
        self.resolved_device = resolve_faiss_device(self.device)
        if self.resolved_device.startswith("cuda") and hasattr(faiss, "StandardGpuResources"):
            gpu_id = int(self.resolved_device.split(":", 1)[1]) if ":" in self.resolved_device else 0
            resources = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(resources, gpu_id, cpu_index)
        else:
            self.index = cpu_index
        self._term_vectors = vectors

    def search(
        self, mentions: list[MentionAnnotation], top_k: int | None = None, threshold: float | None = None
    ) -> list[list[MatchCandidate]]:
        """Retrieve unique ranked codes from the FAISS vector index."""
        faiss = self._require_faiss()
        if self.index is None:
            raise RuntimeError("FAISS index not built")
        k = top_k or self.top_k
        thr = self.threshold if threshold is None else threshold
        overfetch = min(len(self.gazetteer), max(k * 5, k))
        query = self.vectorizer.transform([normalize_text(m.text) for m in mentions]).astype(np.float32).toarray()
        if self.similarity == "cosine":
            faiss.normalize_L2(query)
        scores, indices = self.index.search(query, overfetch)
        results: list[list[MatchCandidate]] = []
        for mention, row_scores, row_indices in zip(mentions, scores, indices):
            cands: list[MatchCandidate] = []
            seen: set[str] = set()
            for score, idx in zip(row_scores, row_indices):
                if idx < 0:
                    continue
                entry = self.gazetteer[int(idx)]
                value = float(score)
                if value < thr:
                    continue
                if self.label_aware and mention.label and entry.label and mention.label != entry.label:
                    continue
                if entry.code in seen:
                    continue
                seen.add(entry.code)
                cands.append(
                    MatchCandidate(
                        None,
                        mention.filename,
                        mention.text,
                        mention.label,
                        entry.code,
                        entry.term,
                        value,
                        "faiss_biencoder",
                        rank=len(cands) + 1,
                        metadata={
                            "device": self.resolved_device,
                            "vectorizer_backend": self.vectorizer_backend,
                            "n_features": self.n_features,
                            "max_features": self.max_features,
                        },
                    )
                )
                if len(cands) >= k:
                    break
            results.append(cands)
        return results
