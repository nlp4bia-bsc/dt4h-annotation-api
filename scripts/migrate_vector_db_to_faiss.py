"""Convert legacy ``.pt`` memmap vector DBs to persisted FAISS indexes.

The old format was a headerless ``numpy.memmap`` of shape ``(n_terms, 768)``
float32 — no metadata, no record of which gazetteer or model produced it.  The
vectors themselves are still fine, so migration reads them straight into a
FAISS ``IndexFlatIP`` and writes a manifest alongside.  No model is loaded and
nothing is re-encoded, which makes this seconds per gazetteer rather than the
minutes a rebuild costs.

Migration is refused, per entry, when the file's size does not match the row
count the gazetteer now resolves to.  That mismatch means the ``.pt`` was built
from a different version of the gazetteer, and its row positions no longer name
the same terms — such an index must be rebuilt, not converted.

Usage
-----
    uv run python scripts/migrate_vector_db_to_faiss.py --dry-run
    uv run python scripts/migrate_vector_db_to_faiss.py [--delete-pt]

The legacy ``.pt`` files are kept unless ``--delete-pt`` is passed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import faiss
import numpy as np

from app.model_manager.resolver import LocalResolver, _make_abs
from app.src.nel import vector_store
from app.src.nel.gazetteer import load_gazetteer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("migrate")

FLOAT32_BYTES = 4
PLAUSIBLE_DIMS = (256, 384, 512, 768, 1024, 1280, 1536)


def infer_dim(pt_path: Path, n_rows: int) -> int:
    """Derive the embedding dimension from file size and row count.

    Raises ``ValueError`` when the size is not an exact multiple, which means
    the file and the current gazetteer disagree about the row count.
    """
    size = pt_path.stat().st_size
    row_bytes = n_rows * FLOAT32_BYTES
    if row_bytes == 0 or size % row_bytes != 0:
        raise ValueError(
            f"{pt_path.name} is {size} bytes, which is not {n_rows} rows x 4 bytes x an "
            "integer dimension. The gazetteer has changed since this file was built."
        )
    dim = size // row_bytes
    if dim not in PLAUSIBLE_DIMS:
        raise ValueError(
            f"{pt_path.name} implies an embedding dimension of {dim}, which is not a "
            "plausible model dimension. Refusing to migrate; rebuild instead."
        )
    return dim


def migrate_one(
    pt_path: Path,
    gaz_path: Path,
    model_path: Path,
    faiss_path: Path,
    *,
    dry_run: bool,
) -> bool:
    """Convert one ``.pt`` file. Returns True when an index was written."""
    rows = load_gazetteer(gaz_path)
    n_rows = len(rows)

    try:
        dim = infer_dim(pt_path, n_rows)
    except ValueError as exc:
        logger.error("SKIP  %s — %s", pt_path.name, exc)
        return False

    if dim != 768:
        logger.warning(
            "%s has dimension %d; the legacy builder only ever wrote 768. "
            "Verify this file before trusting the converted index.",
            pt_path.name, dim,
        )

    if dry_run:
        logger.info("WOULD MIGRATE  %s -> %s (%d rows, dim %d)", pt_path.name, faiss_path.name, n_rows, dim)
        return True

    vectors = np.array(
        np.memmap(pt_path, dtype="float32", mode="r", shape=(n_rows, dim)),
        dtype=np.float32,
        order="C",
    )
    # The legacy builder wrote normalize_embeddings=True vectors, so these are
    # already unit norm; renormalising costs nothing and removes any doubt.
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    vector_store.write_manifest(
        faiss_path,
        gaz_path=gaz_path,
        model_path=model_path,
        dim=dim,
        n_rows=n_rows,
    )
    logger.info("MIGRATED  %s -> %s (%d vectors, dim %d)", pt_path.name, faiss_path.name, n_rows, dim)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report what would happen, write nothing.")
    parser.add_argument(
        "--delete-pt",
        action="store_true",
        help="Delete each legacy .pt after a successful migration. Off by default: "
             "the .pt is the only copy of those embeddings, and rebuilding one costs "
             "a full re-encode of the gazetteer.",
    )
    args = parser.parse_args(argv)

    resolver = LocalResolver()
    entries = (resolver.registry.get("vectorized_dbs") or {})
    if not entries:
        logger.error("No vectorized_dbs entries in the registry — nothing to migrate.")
        return 1

    migrated = skipped = 0

    for lang, tasks in entries.items():
        for entity, raw in (tasks or {}).items():
            if raw is None:
                continue
            pt_path = _make_abs(raw)
            if pt_path.suffix != ".pt":
                continue
            if not pt_path.exists():
                logger.error("SKIP  %s/%s — registered .pt is missing at %s", lang, entity, pt_path)
                skipped += 1
                continue

            try:
                gaz_path = resolver.get_gaz_path(lang, entity)
                model_path, _ = resolver.get_nel_path(lang)
            except Exception as exc:
                logger.error("SKIP  %s/%s — %s", lang, entity, exc)
                skipped += 1
                continue

            faiss_path = pt_path.with_suffix(".faiss")

            try:
                ok = migrate_one(pt_path, gaz_path, model_path, faiss_path, dry_run=args.dry_run)
            except Exception:
                logger.exception("SKIP  %s/%s — migration failed", lang, entity)
                skipped += 1
                continue

            if not ok:
                skipped += 1
                continue

            migrated += 1
            if args.dry_run:
                continue

            tasks[entity] = str(faiss_path)
            resolver.upload_registry()

            if args.delete_pt:
                pt_path.unlink()
                logger.info("Removed legacy %s", pt_path.name)

    logger.info("Done — %d migrated, %d skipped.", migrated, skipped)
    if skipped:
        logger.info(
            "Skipped entries must be rebuilt: delete their registry value and run "
            "'uv run python -m app.model_manager'."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
