"""Gazetteer loading — the single source of truth for vector-DB row order.

A FAISS index stores vectors and returns row positions.  Turning those
positions back into ``(term, code)`` pairs only works if the rows are derived
the same way at build time and at query time.

Before this module the two sides used different code: the builder took
``gaz_df["term"].unique()`` and the query path took
``drop_duplicates(subset=["term"])``.  Those happen to agree today, but nothing
enforced it and nothing would have detected a divergence — the index would
simply have returned the wrong codes.

``load_gazetteer`` is now the only way either side reads a gazetteer, and
``fingerprint`` lets a built index record exactly which file it was built from.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("term", "code")

_HASH_CHUNK_BYTES = 1 << 20


def load_gazetteer(gaz_path: str | Path) -> pd.DataFrame:
    """Read a gazetteer TSV into canonical, positionally-indexed rows.

    The returned frame has exactly the columns ``term`` and ``code``, a
    ``RangeIndex``, and one row per unique term.  Row *i* of this frame
    corresponds to row *i* of the vector DB built from it.

    Deduplication keeps the first row for each term.  A term that appears with
    several different codes therefore resolves to whichever code the gazetteer
    lists first; this preserves the behaviour the pipeline has always had.
    Duplicates are counted and logged so the loss is at least visible.

    Parameters
    ----------
    gaz_path:
        Path to a tab-separated file with ``term`` and ``code`` columns.

    Returns
    -------
    pandas.DataFrame
        Columns ``term`` and ``code``, both ``str``, indexed ``0..n-1``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a required column is missing or no usable rows remain.
    """
    gaz_path = Path(gaz_path)
    if not gaz_path.exists():
        raise FileNotFoundError(f"Gazetteer not found: {gaz_path}")

    # dtype=str is load-bearing, not defensive. SNOMED codes are numeric
    # strings; letting pandas infer the column gives int64, or float64 as soon
    # as one row has a blank code, and str(840539006.0) is "840539006.0".
    frame = pd.read_csv(gaz_path, sep="\t", dtype=str)

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Gazetteer {gaz_path} is missing required column(s) {missing}. "
            f"Found: {list(frame.columns)}"
        )

    frame = frame[list(REQUIRED_COLUMNS)].dropna()
    frame["term"] = frame["term"].astype(str).str.strip()
    frame["code"] = frame["code"].astype(str).str.strip()
    frame = frame[(frame["term"] != "") & (frame["code"] != "")]

    n_before = len(frame)
    frame = frame.drop_duplicates(subset=["term"], keep="first").reset_index(drop=True)
    n_dropped = n_before - len(frame)
    if n_dropped:
        logger.info(
            "Gazetteer %s: dropped %d duplicate term row(s), kept first code for each",
            gaz_path.name, n_dropped,
        )

    if frame.empty:
        raise ValueError(f"Gazetteer {gaz_path} contains no usable term/code rows")

    return frame


def fingerprint(gaz_path: str | Path) -> str:
    """Return the SHA-256 of a gazetteer file, for index manifest validation.

    Hashing the raw bytes rather than the parsed frame means any edit to the
    file — including ones that change row order without changing the row set —
    invalidates the index.  That is the intent: row order *is* the contract.
    """
    digest = hashlib.sha256()
    with Path(gaz_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
