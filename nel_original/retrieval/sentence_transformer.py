from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..utils.device import optional_module, resolve_torch_device


class SentenceTransformerBiEncoder:
    """SentenceTransformer wrapper with GPU-first device resolution and .cuda/.to helpers."""

    def __init__(self, model_or_path: str | Any, device: str = "auto") -> None:
        self.device = resolve_torch_device(device)
        if isinstance(model_or_path, str):
            sentence_transformers = optional_module("sentence_transformers")
            if sentence_transformers is None:
                raise ImportError("sentence-transformers is required for SentenceTransformerBiEncoder")
            self.model = sentence_transformers.SentenceTransformer(model_or_path, device=self.device)
        else:
            self.model = model_or_path
            self.to(self.device)

    def cuda(self, device: int | None = None) -> "SentenceTransformerBiEncoder":
        """Move the encoder to a CUDA device."""
        target = f"cuda:{device}" if device is not None else "cuda"
        self.to(target)
        return self

    def to(self, device: str | Any) -> "SentenceTransformerBiEncoder":
        """Move the encoder to the requested device."""
        self.device = str(device)
        if hasattr(self.model, "to"):
            self.model.to(device)
        return self

    def encode(self, texts: list[str], batch_size: int = 32, normalize_embeddings: bool = True) -> Any:
        """Encode texts as tensors using the wrapped SentenceTransformer."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
            device=self.device,
        )


class DenseRetriever:
    """GPU-first dense retriever backed by SentenceTransformers and torch matrix products."""

    def __init__(
        self,
        df_candidates: pd.DataFrame,
        model_or_path: str | Any,
        normalize: bool = True,
        vector_db: Any | None = None,
        vector_db_batch_size: int = 256,
        device: str = "auto",
    ) -> None:
        self.torch = optional_module("torch")
        if self.torch is None:
            raise ImportError("torch is required for DenseRetriever")
        self.normalize = normalize
        self.device = resolve_torch_device(device)
        self.df_candidates = df_candidates.copy()
        if "term" not in self.df_candidates.columns or "code" not in self.df_candidates.columns:
            raise ValueError("df_candidates must contain term and code columns")

        self.encoder = SentenceTransformerBiEncoder(model_or_path, device=self.device)
        if vector_db is None:
            self.vector_db = self.encoder.encode(
                self.df_candidates["term"].astype(str).tolist(),
                batch_size=vector_db_batch_size,
                normalize_embeddings=self.normalize,
            )
        else:
            self.vector_db = self.normalize_vector(vector_db) if self.normalize else vector_db
            self.vector_db = self.vector_db.to(self.device)

    def cuda(self, device: int | None = None) -> "DenseRetriever":
        """Move the retriever, encoder and vector database to CUDA."""
        target = f"cuda:{device}" if device is not None else "cuda"
        return self.to(target)

    def to(self, device: str | Any) -> "DenseRetriever":
        """Move the retriever, encoder and vector database to a device."""
        self.device = str(device)
        self.encoder.to(device)
        self.vector_db = self.vector_db.to(device)
        return self

    def get_distances(self, data: list[str] | Any, input_format: str = "text") -> tuple[np.ndarray, np.ndarray]:
        """Return the full similarity matrix and descending candidate indices."""
        if input_format == "text":
            q = self.encoder.encode(data, normalize_embeddings=self.normalize)
        elif input_format == "vector":
            q = self.normalize_vector(data.to(self.device)) if self.normalize else data.to(self.device)
        else:
            raise ValueError("input_format must be text or vector")
        sim = self.torch.mm(q, self.vector_db.T)
        distances = sim.detach().cpu().numpy()
        indices = distances.argsort(axis=1)[:, ::-1]
        return distances, indices

    def retrieve_top_k(
        self, data: list[str] | Any, k: int = 10, input_format: str = "text"
    ) -> list[dict[str, list[str | float]]]:
        """Retrieve aligned code, term and similarity lists for each query."""
        distances, indices = self.get_distances(data, input_format=input_format)
        topk = []
        for qi in range(distances.shape[0]):
            codes, terms, similarity = [], [], []
            for idx in indices[qi, :k]:
                codes.append(str(self.df_candidates.iloc[idx]["code"]))
                terms.append(str(self.df_candidates.iloc[idx]["term"]))
                similarity.append(float(distances[qi, idx]))
            topk.append({"codes": codes, "terms": terms, "similarity": similarity})
        return topk

    def normalize_vector(self, vector: Any, p: int | float = 2) -> Any:
        """Normalize a tensor row-wise with the requested norm."""
        norm = self.torch.norm(vector, p=p, dim=1, keepdim=True)
        norm[norm == 0] = 1.0
        return vector / norm
