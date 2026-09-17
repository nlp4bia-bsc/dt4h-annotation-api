"""Legacy .pt memmap → FAISS conversion.

The vectors in a ``.pt`` are still good, so migration reads them straight into
an index rather than re-encoding. The risk is that the gazetteer changed since
the ``.pt`` was written, in which case its row positions name different terms
and the file must be rebuilt, not converted.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.src.nel import vector_store
from scripts.migrate_vector_db_to_faiss import infer_dim, migrate_one

DIM = 768


def write_legacy_pt(path, n_rows: int, dim: int = DIM) -> np.ndarray:
    """Write a headerless float32 memmap of unit-norm vectors, as the old builder did."""
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(n_rows, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    handle = np.memmap(path, dtype="float32", mode="w+", shape=(n_rows, dim))
    handle[:] = vectors
    handle.flush()
    del handle
    return vectors


class EchoEncoder:
    """Returns a stored vector by row index, so top-1 must be that row."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors

    def get_sentence_embedding_dimension(self) -> int:
        return self.vectors.shape[1]

    def encode(self, texts, **kwargs):
        return np.ascontiguousarray(self.vectors[[int(t) for t in texts]], dtype=np.float32)


def test_infer_dim_from_file_size(tmp_path):
    path = tmp_path / "legacy.pt"
    write_legacy_pt(path, n_rows=6)
    assert infer_dim(path, 6) == DIM


def test_infer_dim_rejects_a_row_count_that_does_not_divide(tmp_path):
    path = tmp_path / "legacy.pt"
    write_legacy_pt(path, n_rows=6)
    with pytest.raises(ValueError, match="gazetteer has changed"):
        infer_dim(path, 5)


def test_infer_dim_rejects_an_implausible_dimension(tmp_path):
    path = tmp_path / "legacy.pt"
    write_legacy_pt(path, n_rows=6, dim=7)
    with pytest.raises(ValueError, match="plausible"):
        infer_dim(path, 6)


def test_migration_writes_an_index_and_manifest(tmp_path, gaz_path, model_path):
    pt_path = tmp_path / "disease.pt"
    write_legacy_pt(pt_path, n_rows=6)
    faiss_path = tmp_path / "disease.faiss"

    assert migrate_one(pt_path, gaz_path, model_path, faiss_path, dry_run=False)
    assert faiss_path.exists()
    assert vector_store.manifest_path_for(faiss_path).exists()


def test_migration_preserves_the_original_vectors(tmp_path, gaz_path, model_path):
    pt_path = tmp_path / "disease.pt"
    vectors = write_legacy_pt(pt_path, n_rows=6)
    faiss_path = tmp_path / "disease.faiss"
    migrate_one(pt_path, gaz_path, model_path, faiss_path, dry_run=False)

    store = vector_store.load(faiss_path, gaz_path, model_path)
    encoder = EchoEncoder(vectors)
    hits = store.search(["0", "3", "5"], encoder=encoder, k=1)

    # Querying with row i's own vector must return row i at similarity 1.
    assert [h[0].term for h in hits] == [
        "meningitis bacteriana", "infección por coronavirus", "insuficiencia cardiaca congestiva",
    ]
    assert all(h[0].score == pytest.approx(1.0, abs=1e-4) for h in hits)


def test_the_legacy_file_is_kept(tmp_path, gaz_path, model_path):
    """The .pt is the only copy of those embeddings; rebuilding costs a re-encode."""
    pt_path = tmp_path / "disease.pt"
    write_legacy_pt(pt_path, n_rows=6)
    migrate_one(pt_path, gaz_path, model_path, tmp_path / "disease.faiss", dry_run=False)
    assert pt_path.exists()


def test_dry_run_writes_nothing(tmp_path, gaz_path, model_path):
    pt_path = tmp_path / "disease.pt"
    write_legacy_pt(pt_path, n_rows=6)
    faiss_path = tmp_path / "disease.faiss"

    assert migrate_one(pt_path, gaz_path, model_path, faiss_path, dry_run=True)
    assert not faiss_path.exists()


def test_a_changed_gazetteer_is_refused_not_converted(tmp_path, gaz_path, model_path):
    pt_path = tmp_path / "disease.pt"
    write_legacy_pt(pt_path, n_rows=6)
    gaz_path.write_text(
        gaz_path.read_text(encoding="utf-8") + "gripe\t6142004\n", encoding="utf-8"
    )
    faiss_path = tmp_path / "disease.faiss"

    assert migrate_one(pt_path, gaz_path, model_path, faiss_path, dry_run=False) is False
    assert not faiss_path.exists()
