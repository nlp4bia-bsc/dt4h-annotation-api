from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from app.config import REGISTRY_PATH, RESOURCES_PATH

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    pass


class LocalResolver:
    """
    Single source of truth for:
      - where resources *should* live on disk (naming conventions)
      - whether they already do (existence checks)
      - reading and persisting the registry YAML

    Every path that ModelManager or ResourceDownloader touches must come
    through one of the ``get_*_path`` methods so that creation and access
    always agree on the layout.
    """

    def __init__(self) -> None:
        self.base_pth = Path(RESOURCES_PATH)
        self.reg_path = Path(REGISTRY_PATH)
        self.registry: dict = self._import_registry()

    # ------------------------------------------------------------------
    # Registry I/O
    # ------------------------------------------------------------------

    def _import_registry(self) -> dict:
        """Load registry.yaml, returning an empty dict on any error."""
        try:
            with open(self.reg_path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except FileNotFoundError:
            logger.error("Registry file not found at %s.", self.reg_path)
            return {}
        except yaml.YAMLError as exc:
            logger.error("Failed to parse registry YAML: %s", exc)
            return {}

    def upload_registry(self) -> None:
        """Persist the current in-memory registry back to disk."""
        try:
            with open(self.reg_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(
                    self.registry,
                    stream=fh,
                    default_flow_style=False,
                    sort_keys=False,
                )
        except Exception as exc:
            logger.error("Failed to write registry: %s", exc)

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def get_ner_path(self, lang: str, entity: str) -> tuple[Path, Optional[str]]:
        """
        Returns ``(local_path, repo_id_or_None)``.

        - ``repo_id`` is ``None`` when the model is already present locally.
        - Raises ``ModelNotFoundError`` when the registry entry is incomplete.
        - Raises ``FileNotFoundError`` when a registered path does not exist.
        """
        try:
            cfg = self.registry["ner"][lang][entity]
        except KeyError:
            raise ModelNotFoundError(
                f"No NER entry registered for {lang!r} / {entity!r}."
            )

        pth = cfg.get("local_path")

        if pth is None:
            repo_id = cfg.get("repo_id")
            if not repo_id:
                raise ModelNotFoundError(
                    f"No repo_id provided for NER model {lang!r} / {entity!r}."
                )
            local_path = (
                self.base_pth
                / "local_models"
                / "ner_models"
                / lang
                / entity
                / repo_id.split("/")[-1]
            )
            logger.info(
                "NER model %r / %r not yet downloaded — target: %s",
                lang, entity, local_path,
            )
            return local_path, repo_id

        local_path = Path(pth)
        if not local_path.exists():
            raise FileNotFoundError(
                f"NER model for {lang!r} / {entity!r} not found at {local_path!r} "
                "(path is registered but the directory is missing)."
            )
        return local_path, None

    def get_nel_path(self, lang: str) -> tuple[Path, Optional[str]]:
        """
        Returns ``(local_path, repo_id_or_None)``.

        Semantics mirror ``get_ner_path``.
        """
        try:
            cfg = self.registry["nel"][lang]
        except KeyError:
            raise ModelNotFoundError(f"No NEL entry registered for {lang!r}.")

        pth = cfg.get("local_path")

        if pth is None:
            repo_id = cfg.get("repo_id")
            if not repo_id:
                raise ModelNotFoundError(
                    f"No repo_id provided for NEL model {lang!r}."
                )
            local_path = (
                self.base_pth
                / "local_models"
                / "nel_models"
                / lang
                / repo_id.split("/")[-1]
            )
            logger.info(
                "NEL model %r not yet downloaded — target: %s", lang, local_path
            )
            return local_path, repo_id

        local_path = Path(pth)
        if not local_path.exists():
            raise FileNotFoundError(
                f"NEL model for {lang!r} not found at {local_path!r} "
                "(path is registered but the directory is missing)."
            )
        return local_path, None

    def get_gaz_path(self, lang: str, entity: str) -> Path:
        """
        Returns the validated path to an existing gazetteer file.

        Raises ``ModelNotFoundError`` when the registry entry is absent and
        ``FileNotFoundError`` when the registered file does not exist.
        """
        try:
            raw = self.registry["gazetteers"][lang][entity]
        except KeyError:
            raise ModelNotFoundError(
                f"No gazetteer registered for {lang!r} / {entity!r}. "
                "Add an absolute path to an existing .csv or .tsv file in "
                "app/registry.yaml under gazetteers › <lang> › <entity>."
            )

        pth = Path(raw)
        if not pth.exists():
            raise FileNotFoundError(
                f"Gazetteer for {lang!r} / {entity!r} not found at {pth!r} "
                "(path is registered but the file is missing)."
            )
        return pth

    def _get_nel_model_name(self, lang: str) -> str:
        """Derive the NEL model directory name for path construction.

        Prefers the stem of ``local_path`` when already set; falls back to the
        last segment of ``repo_id`` when the model has not been downloaded yet.
        """
        try:
            cfg = self.registry["nel"][lang]
        except KeyError:
            raise ModelNotFoundError(f"No NEL entry registered for {lang!r}.")

        local_path = cfg.get("local_path")
        if local_path:
            return Path(local_path).name

        repo_id = cfg.get("repo_id")
        if repo_id:
            return repo_id.split("/")[-1]

        raise ModelNotFoundError(
            f"NEL entry for {lang!r} has neither local_path nor repo_id — "
            "cannot derive a model name for the vector DB filename."
        )

    def get_vector_db_path(self, lang: str, entity: str) -> tuple[Path, bool]:
        """
        Returns ``(local_path, already_built)``.

        - ``already_built=True``  → file exists, nothing to do.
        - ``already_built=False`` → path is the *target* for the build step.

        The generated filename embeds the NEL model name so that swapping the
        NEL model automatically produces a new path and triggers a rebuild.
        Format: ``vectorized_dbs/{lang}/{entity}_{nel_model_name}.pt``

        Raises ``ModelNotFoundError`` when the registry key is absent and
        ``FileNotFoundError`` when a non-null registered path does not exist.
        """
        try:
            raw = self.registry["vectorized_dbs"][lang][entity]
        except KeyError:
            raise ModelNotFoundError(
                f"No vectorized_dbs entry for {entity!r} / {lang!r}."
            )

        if raw is None:
            nel_model_name = self._get_nel_model_name(lang)
            target = (
                self.base_pth / "vectorized_dbs" / lang / f"{entity}_{nel_model_name}.pt"
            )
            logger.info(
                "Vector DB for %r / %r not yet built — target: %s",
                lang, entity, target,
            )
            return target, False

        pth = Path(raw)
        if not pth.exists():
            raise FileNotFoundError(
                f"Vector DB for {lang!r} / {entity!r} not found at {pth!r} "
                "(path is registered but the file is missing)."
            )
        return pth, True