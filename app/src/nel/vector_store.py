"""Persistent FAISS vector store for entity linking.

Replaces the raw ``.pt`` memmap + full-matrix ``argsort`` that the biencoder
used previously.  Three things change:

*Persistence.*  An index is built once, written to disk next to a manifest, and
read back on every later run.  Nothing re-encodes a gazetteer that has already
been encoded.

*In-process reuse.*  ``load`` caches by path, so repeated requests in one
process share a single index and a single loaded model rather than re-reading
either from disk.

*Selection.*  FAISS returns the top *k* directly.  The previous path built the
full ``n_mentions x n_gazetteer`` similarity matrix, copied it to CPU as NumPy
and sorted every row end to end in order to take the single best hit.

Index type
----------
``IndexFlatIP`` over L2-normalised embeddings — exact inner product, which on
unit vectors is exactly cosine similarity.  Results are therefore identical to
the previous ``torch.mm`` path, only selected more cheaply.

``IndexFlat`` needs no training.  FAISS's ``index.train()`` step applies to the
quantised and partitioned families (``IVF*``, ``PQ``, ``SQ``), which learn
centroids or quantisation boundaries from a sample before accepting vectors.
Those trade recall for memory and speed and only start paying off in the
millions of vectors; ``FAISS_INDEX_TYPE`` records which family an index on disk
belongs to so a future switch invalidates existing indexes rather than silently
mixing them.

Alignment
---------
FAISS returns row positions, not codes.  Position *i* means "row *i* of the
frame this index was built from", so the index is worthless without the exact
row order used at build time.  Rather than duplicate those rows on disk, the
manifest pins the gazetteer's SHA-256 and both sides derive rows through
``gazetteer.load_gazetteer``.  A gazetteer edited after the index was built
fails the hash check and is reported as needing a rebuild, instead of silently
returning codes belonging to different terms.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.config import device
from app.src.nel.gazetteer import fingerprint, load_gazetteer

logger = logging.getLogger(__name__)

FAISS_INDEX_TYPE = "IndexFlatIP"
MANIFEST_VERSION = 1
ENCODE_CHUNK_SIZE = 10_000
ENCODE_BATCH_SIZE = 256

# Keyed by resolved index path. Both entries are immutable once built, so
# sharing them across requests in one process is safe.
_STORE_CACHE: dict[Path, "FaissVectorStore"] = {}
_MODEL_CACHE: dict[Path, SentenceTransformer] = {}


class VectorStoreError(RuntimeError):
    """An index is missing, stale, or inconsistent with its gazetteer."""


def manifest_path_for(index_path: str | Path) -> Path:
    """Return the manifest path that accompanies *index_path*."""
    index_path = Path(index_path)
    return index_path.with_suffix(index_path.suffix + ".manifest.json")


def get_encoder(model_path: str | Path) -> SentenceTransformer:
    """Load a SentenceTransformer once per process and reuse it thereafter."""
    model_path = Path(model_path).resolve()
    cached = _MODEL_CACHE.get(model_path)
    if cached is not None:
        return cached

    logger.info("Loading NEL encoder: %s", model_path)
    model = SentenceTransformer(str(model_path), device=device)
    _MODEL_CACHE[model_path] = model
    return model


@dataclass(frozen=True)
class SearchHit:
    """One retrieved gazetteer entry for one query mention."""

    code: str
    term: str
    score: float


class FaissVectorStore:
    """A persisted FAISS index plus the gazetteer rows it was built from."""

    def __init__(self, index: faiss.Index, rows: pd.DataFrame, manifest: dict) -> None:
        if index.ntotal != len(rows):
            raise VectorStoreError(
                f"Index holds {index.ntotal} vectors but the gazetteer resolves to "
                f"{len(rows)} rows — the index is stale. Delete it and rebuild."
            )
        self.manifest = manifest
        self._rows = rows
        self._codes: list[str] = rows["code"].tolist()
        self._terms: list[str] = rows["term"].tolist()
        self.index, self.faiss_device = _maybe_to_gpu(index)

    @property
    def dim(self) -> int:
        """Embedding dimension of the indexed vectors."""
        return self.index.d

    def __len__(self) -> int:
        return self.index.ntotal

    def search(
        self,
        queries: Iterable[str],
        encoder: SentenceTransformer,
        k: int = 1,
    ) -> list[list[SearchHit]]:
        """Return the top-*k* gazetteer entries for each query mention.

        Fewer than *k* hits are returned when the gazetteer is smaller than *k*.

        Parameters
        ----------
        queries:
            Mention strings to link.
        encoder:
            The SentenceTransformer the index was built with.  Using a
            different model produces meaningless similarities; the manifest
            records which model was used so callers can check.
        k:
            Number of candidates per mention.
        """
        query_list = list(queries)
        if not query_list:
            return []

        k = max(1, min(k, self.index.ntotal))
        query_vectors = _encode(encoder, query_list, batch_size=ENCODE_BATCH_SIZE)
        scores, indices = self.index.search(query_vectors, k)

        results: list[list[SearchHit]] = []
        for row_scores, row_indices in zip(scores, indices):
            hits: list[SearchHit] = []
            for score, position in zip(row_scores, row_indices):
                # FAISS pads with -1 when it finds fewer neighbours than asked.
                if position < 0:
                    continue
                hits.append(
                    SearchHit(
                        code=self._codes[int(position)],
                        term=self._terms[int(position)],
                        score=float(score),
                    )
                )
            results.append(hits)
        return results


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------


def build(
    gaz_path: str | Path,
    model_path: str | Path,
    index_path: str | Path,
    *,
    chunk_size: int = ENCODE_CHUNK_SIZE,
) -> Path:
    """Encode a gazetteer and write a FAISS index plus its manifest.

    Overwrites any existing index at *index_path*.  Returns the index path.

    The embedding dimension is read from the model rather than assumed, so
    models that are not 768-dimensional (for example the XLM-R *large* SapBERT
    variants, at 1024) build correctly.
    """
    gaz_path, model_path, index_path = Path(gaz_path), Path(model_path), Path(index_path)

    if not model_path.exists():
        raise VectorStoreError(
            f"NEL model not found at {model_path}. Download it before building a vector DB."
        )

    rows = load_gazetteer(gaz_path)
    terms = rows["term"].tolist()
    encoder = get_encoder(model_path)
    dim = encoder.get_sentence_embedding_dimension()

    logger.info(
        "Building FAISS index: %d terms, dim=%d, gaz=%s, model=%s",
        len(terms), dim, gaz_path.name, model_path.name,
    )

    index = faiss.IndexFlatIP(dim)
    for start in range(0, len(terms), chunk_size):
        chunk = terms[start : start + chunk_size]
        index.add(_encode(encoder, chunk, batch_size=ENCODE_BATCH_SIZE))
        logger.debug("Indexed %d/%d terms", min(start + chunk_size, len(terms)), len(terms))

    if index.ntotal != len(terms):
        raise VectorStoreError(
            f"Index built with {index.ntotal} vectors for {len(terms)} terms"
        )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    write_manifest(
        index_path,
        gaz_path=gaz_path,
        model_path=model_path,
        dim=dim,
        n_rows=len(terms),
    )

    logger.info("FAISS index ready: %s (%d vectors)", index_path, index.ntotal)
    return index_path


def write_manifest(
    index_path: str | Path,
    *,
    gaz_path: str | Path,
    model_path: str | Path,
    dim: int,
    n_rows: int,
) -> Path:
    """Write the manifest that binds an index to the gazetteer that built it."""
    index_path = Path(index_path)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "faiss_index_type": FAISS_INDEX_TYPE,
        "normalized": True,
        "embedding_dim": int(dim),
        "n_rows": int(n_rows),
        "nel_model_name": Path(model_path).name,
        "gazetteer_path": str(gaz_path),
        "gazetteer_sha256": fingerprint(gaz_path),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    path = manifest_path_for(index_path)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------


def load(
    index_path: str | Path,
    gaz_path: str | Path,
    model_path: str | Path,
) -> FaissVectorStore:
    """Read a persisted index, validating it against its gazetteer and model.

    Cached per process: the first call reads from disk, later calls with the
    same index path return the same object.

    Raises ``VectorStoreError`` when the index is absent, has no manifest, or
    was built from a different gazetteer, model, or index type.  Every one of
    those would otherwise produce plausible-looking but wrong codes.
    """
    index_path = Path(index_path).resolve()
    cached = _STORE_CACHE.get(index_path)
    if cached is not None:
        return cached

    if not index_path.exists():
        raise VectorStoreError(
            f"FAISS index not found at {index_path} — run 'uv run python -m app.model_manager'"
        )

    manifest = _read_manifest(index_path)
    _validate_manifest(manifest, index_path, gaz_path, model_path)

    rows = load_gazetteer(gaz_path)
    index = faiss.read_index(str(index_path))

    store = FaissVectorStore(index=index, rows=rows, manifest=manifest)
    _STORE_CACHE[index_path] = store
    logger.info(
        "Loaded FAISS index %s (%d vectors, dim=%d, faiss device=%s)",
        index_path.name, len(store), store.dim, store.faiss_device,
    )
    return store


def _read_manifest(index_path: Path) -> dict:
    path = manifest_path_for(index_path)
    if not path.exists():
        raise VectorStoreError(
            f"FAISS index {index_path.name} has no manifest at {path.name}. "
            "Its row alignment cannot be verified — delete the index and rebuild."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VectorStoreError(f"Manifest {path} is not valid JSON: {exc}") from exc


def _validate_manifest(
    manifest: dict,
    index_path: Path,
    gaz_path: str | Path,
    model_path: str | Path,
) -> None:
    rebuild = f"Delete {index_path.name} and rerun 'uv run python -m app.model_manager'."

    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise VectorStoreError(
            f"{index_path.name}: manifest version {manifest.get('manifest_version')!r}, "
            f"expected {MANIFEST_VERSION}. {rebuild}"
        )

    if manifest.get("faiss_index_type") != FAISS_INDEX_TYPE:
        raise VectorStoreError(
            f"{index_path.name}: built as {manifest.get('faiss_index_type')!r}, "
            f"this build expects {FAISS_INDEX_TYPE!r}. {rebuild}"
        )

    expected_model = Path(model_path).name
    if manifest.get("nel_model_name") != expected_model:
        raise VectorStoreError(
            f"{index_path.name}: built with model {manifest.get('nel_model_name')!r}, "
            f"but {expected_model!r} is configured. {rebuild}"
        )

    actual_hash = fingerprint(gaz_path)
    if manifest.get("gazetteer_sha256") != actual_hash:
        raise VectorStoreError(
            f"{index_path.name}: the gazetteer has changed since this index was built "
            f"({Path(gaz_path).name}). Row positions no longer map to the same terms, "
            f"so retrieved codes would be wrong. {rebuild}"
        )


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _encode(encoder: SentenceTransformer, texts: list[str], *, batch_size: int) -> np.ndarray:
    """Encode texts as unit-norm float32 rows, the layout ``IndexFlatIP`` needs.

    ``normalize_embeddings=True`` already returns unit vectors; the explicit
    ``normalize_L2`` guards against a model whose own normalisation leaves
    rounding drift, which would show up as cosine scores slightly above 1.
    """
    vectors = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        device=device,
    )
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)
    return vectors


def _maybe_to_gpu(index: faiss.Index) -> tuple[faiss.Index, str]:
    """Move an index to GPU when the installed FAISS build can reach one.

    Returns the index unchanged on CPU-only installs.  ``faiss-cpu`` has no
    ``StandardGpuResources`` at all, and a GPU build on a machine with no
    visible device reports ``get_num_gpus() == 0``; both are checked, because
    testing only for the attribute would pass on the second case and then fail
    inside FAISS.
    """
    if device != "cuda":
        return index, "cpu"

    has_gpu_api = hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu")
    if not has_gpu_api:
        logger.info("FAISS CPU build installed — index stays on CPU. For GPU: uv sync --extra gpu")
        return index, "cpu"

    if not (hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0):
        logger.info("FAISS GPU build installed but no GPU visible — index stays on CPU")
        return index, "cpu"

    try:
        resources = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(resources, 0, index)
    except Exception:
        logger.warning("Could not move FAISS index to GPU — continuing on CPU", exc_info=True)
        return index, "cpu"

    # Keep the resources alive: FAISS does not own them and the index segfaults
    # if they are garbage collected while it is still in use.
    gpu_index._gpu_resources = resources
    return gpu_index, "cuda:0"


def clear_cache() -> None:
    """Drop cached indexes and encoders. Intended for tests and migrations."""
    _STORE_CACHE.clear()
    _MODEL_CACHE.clear()
