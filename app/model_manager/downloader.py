from __future__ import annotations

import csv
import gc
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from app.utils.download_model import create_vector_db
from app.config import device

logger = logging.getLogger(__name__)


class ResourceDownloader:
    """
    Handles validation and acquisition of all model assets:
    gazetteers, HuggingFace models (NER / NEL), and vector databases.

    Intended to be called once at setup time via ``ModelManager.sanitize()``.
    It has no registry awareness — callers are responsible for passing
    resolved paths and persisting results back to the registry.
    """

    # ------------------------------------------------------------------
    # Gazetteers
    # ------------------------------------------------------------------

    def check_gazetteer(self, gaz_path: Path) -> str:
        """
        Validate that *gaz_path* exists and contains the required columns
        (``term``, ``code``).  If the file is a CSV it is transparently
        converted to TSV in-place and the new path is returned.

        Returns the (possibly updated) path as a ``str`` so callers can
        persist it directly to the registry.
        """
        required_cols = {"term", "code"}

        if gaz_path.suffix == ".csv":
            tsv_path = gaz_path.with_suffix(".tsv")
            logger.info("Converting CSV gazetteer to TSV: %s → %s", gaz_path, tsv_path)
            with open(gaz_path, newline="") as f_in, open(tsv_path, "w", newline="") as f_out:
                f_out.write(f_in.read().replace(",", "\t"))
            gaz_path = tsv_path

        with open(gaz_path, newline="") as fh:
            headers = set(next(csv.reader(fh, delimiter="\t")))

        missing = required_cols - headers
        if missing:
            raise ValueError(
                f"Gazetteer at {gaz_path!r} is missing required column(s): {missing}."
            )

        logger.info("Gazetteer OK: %s", gaz_path)
        return str(gaz_path)

    # ------------------------------------------------------------------
    # HuggingFace models (NER / NEL)
    # ------------------------------------------------------------------

    def download_hf(self, model_local_path: Path, repo_id: str) -> str:
        """
        Download *repo_id* from HuggingFace Hub into *model_local_path*.

        The parent directory is created if it does not exist.
        Returns the local path as a ``str`` for registry persistence.
        """
        model_local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %r → %s", repo_id, model_local_path)
        snapshot_download(repo_id=repo_id, local_dir=model_local_path)
        logger.info("Download complete: %s", model_local_path)
        return str(model_local_path)

    # ------------------------------------------------------------------
    # Vector databases
    # ------------------------------------------------------------------

    def _get_gaz_terms(self, gaz_pth: Path) -> list[str]:
        """Read unique ``term`` values from a TSV gazetteer."""
        gaz_df = pd.read_csv(gaz_pth, sep="\t")
        terms = list(gaz_df["term"].unique())
        del gaz_df
        gc.collect()
        return terms

    def build_vector_db(
        self,
        gaz_pth: Path,
        nel_local_path: Path,
        vector_db_pth: Path,
    ) -> str:
        """
        Encode gazetteer terms with the NEL sentence-transformer and write
        the resulting tensor database to *vector_db_pth*.

        Returns the path as a ``str`` for registry persistence.

        Raises ``ValueError`` if the NEL model directory does not exist yet
        (the NEL model must be downloaded before vector DBs can be built).
        """
        if not nel_local_path.exists():
            raise ValueError(
                f"NEL model not found at {nel_local_path!r}. "
                "Download the NEL model before building vector databases."
            )

        logger.info(
            "Building vector DB: gaz=%s  nel=%s  out=%s",
            gaz_pth, nel_local_path, vector_db_pth,
        )

        nel_model = SentenceTransformer(str(nel_local_path), device=device)
        vector_db_pth.parent.mkdir(parents=True, exist_ok=True)

        gaz_terms = self._get_gaz_terms(gaz_pth)
        create_vector_db(gaz_terms, nel_model, vector_db_pth)

        del gaz_terms
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        time.sleep(1)

        logger.info("Vector DB ready: %s", vector_db_pth)
        return str(vector_db_pth)