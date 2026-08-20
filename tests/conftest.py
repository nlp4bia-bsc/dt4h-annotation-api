"""Shared fixtures for the NEL test suite.

No test here downloads a model or touches the network. The NEL encoder is
replaced by a deterministic stand-in, so the suite exercises the real FAISS,
gazetteer, fusion and formatter code paths while staying fast and runnable on
a machine with no ``registry.yaml`` and no ``app/resources/``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.src.nel import lexical_index, vector_store  # noqa: E402

EMBEDDING_DIM = 16


class StubEncoder:
    """Stands in for a SentenceTransformer with reproducible embeddings.

    Texts are embedded as unit vectors seeded from a SHA-256 of the normalised
    text. Python's built-in ``hash`` is salted per process, which would make
    any ranking assertion pass or fail depending on ``PYTHONHASHSEED``; hashing
    explicitly keeps the suite deterministic across runs.

    ``overrides`` pins chosen texts to exact vectors, which is how a test says
    "make the dense retriever get this one wrong" without relying on luck.
    """

    def __init__(self, overrides: dict[str, np.ndarray] | None = None) -> None:
        self.overrides = overrides or {}

    def get_sentence_embedding_dimension(self) -> int:
        return EMBEDDING_DIM

    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            key = " ".join(str(text).split()).lower()
            override = self.overrides.get(key)
            if override is not None:
                vectors[row] = override
                continue
            seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
            vectors[row] = np.random.default_rng(seed).normal(size=EMBEDDING_DIM)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def basis_vector(position: int) -> np.ndarray:
    """A one-hot vector, for tests that need exactly controlled similarities."""
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[position % EMBEDDING_DIM] = 1.0
    return vector


@pytest.fixture(autouse=True)
def _clear_module_caches():
    """Isolate tests from each other.

    ``vector_store`` and ``lexical_index`` both cache by path at module level.
    Two tests using the same tmp path would otherwise share state, and a test
    could pass only because an earlier one had populated the cache.
    """
    vector_store.clear_cache()
    lexical_index.clear_cache()
    yield
    vector_store.clear_cache()
    lexical_index.clear_cache()


GAZETTEER_ROWS = [
    ("meningitis bacteriana", "7180009"),
    ("meningitis vírica", "7180009"),  # same concept, second surface form
    ("COVID-19", "840539006"),
    ("infección por coronavirus", "840539006"),
    ("varicela", "38907003"),
    ("insuficiencia cardiaca congestiva", "42343007"),
]


@pytest.fixture
def gaz_path(tmp_path: Path) -> Path:
    """A small, well-formed TSV gazetteer with repeated codes."""
    path = tmp_path / "disease.tsv"
    lines = ["term\tcode"] + [f"{term}\t{code}" for term, code in GAZETTEER_ROWS]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """A stand-in NEL model directory. Only its name reaches the manifest."""
    path = tmp_path / "ClinLinker-KB-P"
    path.mkdir()
    return path


@pytest.fixture
def encoder(model_path: Path) -> StubEncoder:
    """Register a stub encoder so ``get_encoder`` returns it instead of loading."""
    stub = StubEncoder()
    vector_store._MODEL_CACHE[model_path.resolve()] = stub
    return stub


@pytest.fixture
def faiss_index_path(tmp_path: Path, gaz_path: Path, model_path: Path, encoder) -> Path:
    """A built and persisted FAISS index over ``gaz_path``."""
    index_path = tmp_path / "disease_ClinLinker-KB-P.faiss"
    vector_store.build(gaz_path=gaz_path, model_path=model_path, index_path=index_path)
    return index_path


@pytest.fixture
def lexical_index_path(tmp_path: Path) -> Path:
    """Target path for the sparse lexical index; built lazily by the code."""
    return tmp_path / "disease.lexical.pkl"
