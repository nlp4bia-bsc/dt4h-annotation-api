"""Persisted FAISS store: build, reload, cache, and the staleness guards.

The guards matter more than the happy path. FAISS returns row positions, so an
index loaded against a gazetteer it was not built from returns codes belonging
to different terms — plausible output, silently wrong. Every test below that
asserts a refusal is protecting against exactly that.
"""

from __future__ import annotations

import json

import pytest

from app.src.nel import vector_store
from app.src.nel.vector_store import VectorStoreError


def test_build_writes_index_and_manifest(gaz_path, model_path, faiss_index_path):
    assert faiss_index_path.exists()
    manifest_path = vector_store.manifest_path_for(faiss_index_path)
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["n_rows"] == 6
    assert manifest["embedding_dim"] == 16
    assert manifest["nel_model_name"] == "ClinLinker-KB-P"
    assert manifest["faiss_index_type"] == "IndexFlatIP"
    assert manifest["normalized"] is True


def test_embedding_dim_comes_from_the_model_not_a_constant(gaz_path, model_path, faiss_index_path):
    """The pre-FAISS code hardcoded 768, so 1024-dim models could not build."""
    store = vector_store.load(faiss_index_path, gaz_path, model_path)
    assert store.dim == 16


def test_reload_from_disk_returns_the_indexed_rows(gaz_path, model_path, faiss_index_path, encoder):
    vector_store.clear_cache()
    vector_store._MODEL_CACHE[model_path.resolve()] = encoder

    store = vector_store.load(faiss_index_path, gaz_path, model_path)
    assert len(store) == 6

    hits = store.search(["COVID-19"], encoder=encoder, k=1)
    assert hits[0][0].code == "840539006"
    assert hits[0][0].term == "COVID-19"
    assert hits[0][0].score == pytest.approx(1.0, abs=1e-4)


def test_search_returns_distinct_rows_ranked(gaz_path, model_path, faiss_index_path, encoder):
    store = vector_store.load(faiss_index_path, gaz_path, model_path)
    hits = store.search(["meningitis bacteriana"], encoder=encoder, k=3)[0]
    assert len(hits) == 3
    assert hits[0].code == "7180009"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_k_is_capped_at_the_index_size(gaz_path, model_path, faiss_index_path, encoder):
    store = vector_store.load(faiss_index_path, gaz_path, model_path)
    hits = store.search(["covid"], encoder=encoder, k=999)[0]
    assert len(hits) == 6


def test_empty_query_list_returns_empty(gaz_path, model_path, faiss_index_path, encoder):
    store = vector_store.load(faiss_index_path, gaz_path, model_path)
    assert store.search([], encoder=encoder, k=5) == []


def test_load_is_cached_by_path(gaz_path, model_path, faiss_index_path):
    first = vector_store.load(faiss_index_path, gaz_path, model_path)
    second = vector_store.load(faiss_index_path, gaz_path, model_path)
    assert first is second


def test_missing_index_names_the_build_command(tmp_path, gaz_path, model_path):
    with pytest.raises(VectorStoreError, match="app.model_manager"):
        vector_store.load(tmp_path / "absent.faiss", gaz_path, model_path)


def test_missing_manifest_refuses(gaz_path, model_path, faiss_index_path):
    vector_store.manifest_path_for(faiss_index_path).unlink()
    vector_store.clear_cache()
    with pytest.raises(VectorStoreError, match="manifest"):
        vector_store.load(faiss_index_path, gaz_path, model_path)


def test_changed_gazetteer_refuses_rather_than_shifting_rows(gaz_path, model_path, faiss_index_path):
    gaz_path.write_text(
        "term\tcode\ngripe\t6142004\n" + gaz_path.read_text(encoding="utf-8").split("\n", 1)[1],
        encoding="utf-8",
    )
    vector_store.clear_cache()
    with pytest.raises(VectorStoreError, match="gazetteer has changed"):
        vector_store.load(faiss_index_path, gaz_path, model_path)


def test_different_model_refuses(tmp_path, gaz_path, model_path, faiss_index_path):
    other_model = tmp_path / "SapBERT-XLMR-large"
    other_model.mkdir()
    vector_store.clear_cache()
    with pytest.raises(VectorStoreError, match="built with model"):
        vector_store.load(faiss_index_path, gaz_path, other_model)


def test_unknown_index_type_refuses(gaz_path, model_path, faiss_index_path):
    manifest_path = vector_store.manifest_path_for(faiss_index_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["faiss_index_type"] = "IVFPQ"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    vector_store.clear_cache()
    with pytest.raises(VectorStoreError, match="IndexFlatIP"):
        vector_store.load(faiss_index_path, gaz_path, model_path)


def test_manifest_version_mismatch_refuses(gaz_path, model_path, faiss_index_path):
    manifest_path = vector_store.manifest_path_for(faiss_index_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_version"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    vector_store.clear_cache()
    with pytest.raises(VectorStoreError, match="manifest version"):
        vector_store.load(faiss_index_path, gaz_path, model_path)


def test_build_refuses_a_missing_model_directory(tmp_path, gaz_path):
    with pytest.raises(VectorStoreError, match="NEL model not found"):
        vector_store.build(
            gaz_path=gaz_path,
            model_path=tmp_path / "not-downloaded",
            index_path=tmp_path / "out.faiss",
        )


def test_get_encoder_caches_by_path(model_path, encoder):
    assert vector_store.get_encoder(model_path) is encoder
