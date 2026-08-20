"""Transformer bi-encoder retrieval backed by FAISS."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..utils.device import optional_module

logger = logging.getLogger(__name__)


class HerbertFaissBiEncoder:
    """Refined HERBERT-style transformer + FAISS candidate retriever.

    The default ``FlatIP``/``FlatL2`` path intentionally mirrors the original
    HERBERT ``FaissEncoder`` contract: encode ``vocab["term"]`` with a
    HuggingFace ``AutoModel`` using mean pooling over ``last_hidden_state``, add
    vectors to a FAISS ``IndexIDMap`` keyed by vocabulary row id, then retrieve
    unique ``code`` values for each query mention. The implementation is more
    structured and GPU-aware while keeping that retrieval behavior as the
    baseline.

    Additional FAISS index types are opt-in for efficiency experiments:
    ``SQ8``/``SQ4`` scalar quantization, ``IVFFlatIP``/``IVFFlatL2`` and
    ``IVFSQ8``/``IVFPQ`` approximate+compressed variants.
    """

    _ALIASES = {
        "flatip": "flatip",
        "ip": "flatip",
        "inner_product": "flatip",
        "flatl2": "flatl2",
        "l2": "flatl2",
        "sq8": "sq8",
        "scalarquantizer": "sq8",
        "scalar_quantizer": "sq8",
        "quantized": "sq8",
        "quantizedip": "sq8",
        "sq4": "sq4",
        "ivfflatip": "ivfflatip",
        "ivf_flat_ip": "ivfflatip",
        "ivf32,flat": "ivfflatip",
        "ivfflatl2": "ivfflatl2",
        "ivf_flat_l2": "ivfflatl2",
        "ivfsq8": "ivfsq8",
        "ivfsq8ip": "ivfsq8",
        "ivf_sq8_ip": "ivfsq8",
        "ivf32,sq8": "ivfsq8",
        "ivfpq": "ivfpq",
        "ivf_pq": "ivfpq",
        "ivf32,pq": "ivfpq",
    }

    def __init__(
        self,
        model_name_or_path: str,
        f_type: str = "FlatIP",
        max_length: int | None = None,
        vocab: Any | None = None,
        batch_size: int = 64,
        device: str = "auto",
        verbose: int = 0,
        nlist: int = 100,
        nprobe: int = 10,
        pq_m: int = 8,
        pq_bits: int = 8,
        overfetch_factor: int = 5,
        use_gpu: bool = True,
        pooling: str = "mean",
    ) -> None:
        transformers = optional_module("transformers")
        torch = optional_module("torch")
        if transformers is None or torch is None:
            raise ImportError("transformers and torch are required for HerbertFaissBiEncoder")
        self.faiss = self._require_faiss()
        self.torch = torch
        self.model_name_or_path = model_name_or_path
        self.method = f"transformer_faiss_{self.normalize_f_type(f_type)}"
        self.model = transformers.AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        self.f_type = f_type
        self.index_kind = self.normalize_f_type(f_type)
        self.max_length = max_length or self.tokenizer.model_max_length
        self.batch_size = batch_size
        self.verbose = verbose
        self.nlist = nlist
        self.nprobe = nprobe
        self.pq_m = pq_m
        self.pq_bits = pq_bits
        self.overfetch_factor = max(1, overfetch_factor)
        self.use_gpu = use_gpu
        self.pooling = pooling
        self.requested_device = device
        self.device = self._resolve_torch_device(device)
        if self.device.startswith("cuda") and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()
        self.vocab = None
        self.faiss_index: Any | None = None
        self.resolved_faiss_device = "cpu"
        self._gpu_resources: Any | None = None
        if vocab is not None:
            self.set_vocab(vocab)

    @classmethod
    def supported_index_types(cls) -> tuple[str, ...]:
        """Return the public FAISS configuration names accepted by ``f_type``."""
        return ("FlatIP", "FlatL2", "SQ8", "SQ4", "IVFFlatIP", "IVFFlatL2", "IVFSQ8", "IVFPQ")

    @classmethod
    def normalize_f_type(cls, f_type: str) -> str:
        """Normalize user-facing FAISS names to internal index keys."""
        key = f_type.replace(" ", "").replace("-", "_").lower()
        if key not in cls._ALIASES:
            supported = ", ".join(cls.supported_index_types())
            raise ValueError(f"Unsupported f_type '{f_type}'. Supported values: {supported}")
        return cls._ALIASES[key]

    def _require_faiss(self) -> Any:
        faiss = optional_module("faiss")
        if faiss is None:
            raise ImportError("faiss-gpu (preferred) or faiss-cpu is required for HerbertFaissBiEncoder")
        return faiss

    def _resolve_torch_device(self, device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if self.torch.cuda.is_available() else "cpu"

    def set_vocab(self, vocab: Any) -> None:
        """Set the FAISS vocabulary dataframe with required ``term`` and ``code`` columns."""
        if not {"term", "code"}.issubset(vocab.columns):
            raise ValueError("vocab must contain 'term' and 'code' columns")
        self.vocab = vocab[["term", "code"]].dropna().copy().reset_index(drop=True)
        self.vocab["term"] = self.vocab["term"].astype(str)
        self.vocab["code"] = self.vocab["code"].astype(str)
        self.arr_text = self.vocab["term"].tolist()
        self.arr_codes = self.vocab["code"].tolist()
        self.arr_text_id = np.arange(len(self.vocab), dtype=np.int64)

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Encode texts with the transformer model using HERBERT-compatible pooling."""
        if not texts:
            raise ValueError("texts cannot be empty")
        batch_size = batch_size or self.batch_size
        model = self.model.module if hasattr(self.model, "module") else self.model
        embeddings = []
        iterator = range(0, len(texts), batch_size)
        tqdm_mod = optional_module("tqdm.auto") if self.verbose else None
        if tqdm_mod is not None:
            iterator = tqdm_mod.tqdm(iterator, desc="Encoding texts", disable=self.verbose == 0)
        for start in iterator:
            batch_texts = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = self._pool_outputs(outputs.last_hidden_state, inputs)
            embeddings.append(batch_embeddings.detach().cpu().numpy())
        return np.vstack(embeddings).astype("float32")

    def _pool_outputs(self, last_hidden_state: Any, inputs: dict[str, Any]) -> Any:
        if self.pooling == "mean":
            return last_hidden_state.mean(dim=1)
        if self.pooling in {"attention_mask_mean", "masked_mean"}:
            mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size()).float()
            summed = (last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return summed / counts
        if self.pooling == "cls":
            return last_hidden_state[:, 0]
        raise ValueError("pooling must be one of: mean, attention_mask_mean, cls")

    def fit_faiss(self, vocab: Any | None = None, batch_size: int | None = None) -> None:
        """Encode the vocabulary and fit/add a FAISS index.

        ``FlatIP``/``FlatL2`` preserve the original HERBERT behavior; quantized
        and IVF variants train the required FAISS quantizers before adding IDs.
        """
        if vocab is not None:
            self.set_vocab(vocab)
        if self.vocab is None:
            raise ValueError("vocab is required before fitting FAISS")
        batch_size = batch_size or self.batch_size
        embeddings = self.encode(self.arr_text, batch_size=batch_size).astype("float32", copy=False)
        embeddings = self._prepare_index_embeddings(embeddings)
        cpu_index = self._build_cpu_index(embeddings)
        id_index = (
            self.faiss.IndexIDMap2(cpu_index)
            if hasattr(self.faiss, "IndexIDMap2")
            else self.faiss.IndexIDMap(cpu_index)
        )
        id_index.add_with_ids(embeddings, self.arr_text_id)
        self.faiss_index = self._maybe_move_to_gpu(id_index)

    def _uses_inner_product(self) -> bool:
        return self.index_kind in {"flatip", "sq8", "sq4", "ivfflatip", "ivfsq8", "ivfpq"}

    def _prepare_index_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if self._uses_inner_product():
            self.faiss.normalize_L2(embeddings)
        return embeddings

    def _prepare_query_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if self._uses_inner_product():
            self.faiss.normalize_L2(embeddings)
        return embeddings

    def _build_cpu_index(self, embeddings: np.ndarray) -> Any:
        dim = embeddings.shape[1]
        n_train = len(embeddings)
        if self.index_kind == "flatip":
            return self.faiss.IndexFlatIP(dim)
        if self.index_kind == "flatl2":
            return self.faiss.IndexFlatL2(dim)
        if self.index_kind in {"sq8", "sq4"}:
            index = self.faiss.IndexScalarQuantizer(
                dim, self._scalar_quantizer_type(self.index_kind), self.faiss.METRIC_INNER_PRODUCT
            )
            index.train(embeddings)
            return index
        if self.index_kind in {"ivfflatip", "ivfflatl2"}:
            metric = self.faiss.METRIC_INNER_PRODUCT if self.index_kind == "ivfflatip" else self.faiss.METRIC_L2
            quantizer = self.faiss.IndexFlatIP(dim) if self.index_kind == "ivfflatip" else self.faiss.IndexFlatL2(dim)
            index = self.faiss.IndexIVFFlat(quantizer, dim, self._safe_nlist(n_train), metric)
            index.train(embeddings)
            index.nprobe = min(self.nprobe, index.nlist)
            return index
        if self.index_kind == "ivfsq8":
            quantizer = self.faiss.IndexFlatIP(dim)
            index = self.faiss.IndexIVFScalarQuantizer(
                quantizer,
                dim,
                self._safe_nlist(n_train),
                self.faiss.ScalarQuantizer.QT_8bit,
                self.faiss.METRIC_INNER_PRODUCT,
            )
            index.train(embeddings)
            index.nprobe = min(self.nprobe, index.nlist)
            return index
        if self.index_kind == "ivfpq":
            quantizer = self.faiss.IndexFlatIP(dim)
            pq_m = self._safe_pq_m(dim)
            index = self.faiss.IndexIVFPQ(
                quantizer,
                dim,
                self._safe_nlist(n_train),
                pq_m,
                self.pq_bits,
                self.faiss.METRIC_INNER_PRODUCT,
            )
            index.train(embeddings)
            index.nprobe = min(self.nprobe, index.nlist)
            return index
        raise ValueError(f"Unsupported normalized f_type: {self.index_kind}")

    def _scalar_quantizer_type(self, kind: str) -> Any:
        if kind == "sq8":
            return self.faiss.ScalarQuantizer.QT_8bit
        if hasattr(self.faiss.ScalarQuantizer, "QT_4bit"):
            return self.faiss.ScalarQuantizer.QT_4bit
        raise ValueError("SQ4 requires a FAISS build exposing ScalarQuantizer.QT_4bit; use SQ8 instead")

    def _safe_nlist(self, n_train: int) -> int:
        return max(1, min(self.nlist, n_train))

    def _safe_pq_m(self, dim: int) -> int:
        if dim % self.pq_m == 0:
            return self.pq_m
        for candidate in range(min(self.pq_m, dim), 0, -1):
            if dim % candidate == 0:
                return candidate
        return 1

    def _maybe_move_to_gpu(self, index: Any) -> Any:
        self.resolved_faiss_device = "cpu"
        if not self.use_gpu or self.requested_device == "cpu":
            return index

        gpu_count = 0
        if hasattr(self.faiss, "get_num_gpus"):
            gpu_count = int(self.faiss.get_num_gpus())
        gpu_api_available = hasattr(self.faiss, "StandardGpuResources") and hasattr(self.faiss, "index_cpu_to_gpu")
        if not gpu_api_available or gpu_count < 1:
            if self.requested_device.startswith("cuda"):
                raise RuntimeError(
                    "CUDA was requested, but the installed FAISS build cannot access a GPU. "
                    "Install faiss-gpu and verify faiss.get_num_gpus()."
                )
            logger.info("FAISS GPU is unavailable; keeping the index on CPU")
            return index

        requested_gpu_id = 0
        if self.requested_device.startswith("cuda:"):
            requested_gpu_id = int(self.requested_device.split(":", 1)[1])
        if requested_gpu_id >= gpu_count:
            raise RuntimeError(f"FAISS GPU {requested_gpu_id} was requested, but only {gpu_count} GPU(s) are available")

        self._gpu_resources = self.faiss.StandardGpuResources()
        self.resolved_faiss_device = f"cuda:{requested_gpu_id}"
        return self.faiss.index_cpu_to_gpu(self._gpu_resources, requested_gpu_id, index)

    def get_candidates(
        self, texts: list[str], k: int = 200, batch_size: int | None = None
    ) -> tuple[list[list[str]], list[list[str]], list[list[float]]]:
        """Return top-k unique candidate terms, codes and FAISS scores for each text."""
        if self.faiss_index is None:
            raise AttributeError("FAISS index is not initialized. Run fit_faiss first.")
        query_embeddings = self.encode(texts, batch_size=batch_size).astype("float32", copy=False)
        query_embeddings = self._prepare_query_embeddings(query_embeddings)
        overfetch = min(len(self.arr_text), max(k * self.overfetch_factor, k))
        similarities, indices = self.faiss_index.search(query_embeddings, overfetch)
        return self._process_results(indices, similarities, k)

    def _process_results(
        self, indices: np.ndarray, similarities: np.ndarray, k: int
    ) -> tuple[list[list[str]], list[list[str]], list[list[float]]]:
        all_candidates, all_codes, all_scores = [], [], []
        for idx_list, score_list in zip(indices, similarities):
            seen = set()
            candidates, codes, scores = [], [], []
            for idx, score in zip(idx_list, score_list):
                if idx < 0:
                    continue
                code = self.arr_codes[int(idx)]
                if code in seen:
                    continue
                seen.add(code)
                candidates.append(self.arr_text[int(idx)])
                codes.append(code)
                scores.append(float(score))
                if len(codes) >= k:
                    break
            all_candidates.append(candidates)
            all_codes.append(codes)
            all_scores.append(scores)
        return all_candidates, all_codes, all_scores
