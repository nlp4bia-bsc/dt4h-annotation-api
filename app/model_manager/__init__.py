"""
model_manager
=============

Public API
----------
    ModelManager   – orchestrates resource discovery, download, and registry persistence

Internal modules (not re-exported):
    .downloader    – ResourceDownloader (HuggingFace pull, gazetteer check, vector-DB build)
    .resolver      – LocalResolver (path resolution + registry I/O)

Download rules (mirrors registry schema):
    gazetteers     – always validate; no repo_id needed
    ner / nel      – download when repo_id is set AND local_path is absent
    vectorized_dbs – build when the registry value is null

Usage
-----
    from app.model_manager import ModelManager

    manager = ModelManager(device="cuda")
    manager.sanitize()          # safe to call at every startup; no-ops when all resources are present
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

from .downloader import ResourceDownloader
from .resolver import LocalResolver

__all__ = ["ModelManager"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed shape for a pending-download record
# ---------------------------------------------------------------------------

class PendingResource(TypedDict):
    resource: str               # "ner" | "nel" | "gazetteers" | "vectorized_dbs"
    lang: str
    task: str | None        # sub-task key, or None for nel entries
    repo_id: str | None     # None for gazetteers and vectorized_dbs
    local_path: Path        # resolved target path on disk
    registry_keys: tuple    # full key path for update_registry


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------

class ModelManager:
    """
    Orchestrates resource validation and download for all registry entries.

    """

    def __init__(self) -> None:
        self.resolver = LocalResolver()
        self.downloader = ResourceDownloader()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def find_pending_resources(self) -> list[PendingResource]:
        """
        Inspect the registry and return every resource that still needs to be
        downloaded, validated, or built.

        Entries are returned in dependency order:
        gazetteers → ner → nel → vectorized_dbs
        (vector DBs require both a gazetteer and a NEL model.)
        """
        pending: list[PendingResource] = []
        registry = self.resolver.registry

        # --- gazetteers: always validate (they are user-supplied, never downloaded) ---
        for lang, entities in (registry.get("gazetteers") or {}).items():
            for entity in (entities or {}):
                pending.append(
                    PendingResource(
                        resource="gazetteers",
                        lang=lang,
                        task=entity,
                        repo_id=None,
                        local_path=self.resolver.get_gaz_path(lang, entity),
                        registry_keys=("gazetteers", lang, entity),
                    )
                )

        # --- ner: repo_id present AND local_path absent ---
        for lang, tasks in (registry.get("ner") or {}).items():
            for task, cfg in (tasks or {}).items():
                if not (cfg and cfg.get("repo_id") and not cfg.get("local_path")):
                    continue
                local_path, _ = self.resolver.get_ner_path(lang, task)
                pending.append(
                    PendingResource(
                        resource="ner",
                        lang=lang,
                        task=task,
                        repo_id=cfg["repo_id"],
                        local_path=local_path,
                        registry_keys=("ner", lang, task),
                    )
                )

        # --- nel: same rule as ner ---
        for lang, cfg in (registry.get("nel") or {}).items():
            if not (cfg and cfg.get("repo_id") and not cfg.get("local_path")):
                continue
            local_path, _ = self.resolver.get_nel_path(lang)
            pending.append(
                PendingResource(
                    resource="nel",
                    lang=lang,
                    task=None,
                    repo_id=cfg["repo_id"],
                    local_path=local_path,
                    registry_keys=("nel", lang),
                )
            )

        # --- vectorized_dbs: build when registry value is null ---
        for lang, tasks in (registry.get("vectorized_dbs") or {}).items():
            for task in (tasks or {}):
                local_path, already_built = self.resolver.get_vector_db_path(lang, task)
                if not already_built:
                    pending.append(
                        PendingResource(
                            resource="vectorized_dbs",
                            lang=lang,
                            task=task,
                            repo_id=None,
                            local_path=local_path,
                            registry_keys=("vectorized_dbs", lang, task),
                        )
                    )

        return pending

    # ------------------------------------------------------------------
    # Registry persistence
    # ------------------------------------------------------------------

    def update_registry(self, registry_keys: tuple, local_path: str | Path) -> None:
        """
        Write a successfully resolved *local_path* back into the in-memory
        registry, then persist it to disk.

        ``registry_keys`` is the full path from the root to the target leaf:

            ("ner", "es", "disease")        → registry["ner"]["es"]["disease"]["local_path"]
            ("nel", "es")                   → registry["nel"]["es"]["local_path"]
            ("gazetteers", "es", "disease") → registry["gazetteers"]["es"]["disease"]
            ("vectorized_dbs", "es", "disease") → registry["vectorized_dbs"]["es"]["disease"]

        For ner / nel the leaf value is a dict with a ``local_path`` key.
        For gazetteers / vectorized_dbs the leaf value is a plain string.
        """
        node = self.resolver.registry

        for key in registry_keys[:-1]:
            node = node[key]

        leaf_key = registry_keys[-1]
        str_path = str(local_path)

        if isinstance(node.get(leaf_key), dict):
            node[leaf_key]["local_path"] = str_path
        else:
            node[leaf_key] = str_path

        self.resolver.upload_registry()
        logger.debug("Registry updated: %s → %s", " › ".join(registry_keys), str_path)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def sanitize(self) -> None:
        """
        Run the full setup pipeline:

        1. Discover all resources that need attention.
        2. For each resource, validate / download / build as appropriate.
        3. Persist the registry after every successful step so partial
           progress is not lost on failure.

        Resources that fail are logged as errors and skipped; the pipeline
        continues with the remaining items so a single bad entry does not
        block everything else.

        Safe to call on every startup — exits immediately when all resources
        are already present.
        """
        pending = self.find_pending_resources()

        if not pending:
            logger.info("All resources are already present — nothing to do.")
            return

        logger.info("Found %d resource(s) to process.", len(pending))
        errors: list[str] = []

        for item in pending:
            label = self._label(item)
            local_path: Path = item["local_path"]
            repo_id = item["repo_id"]

            logger.info(
                "[%s]  %s",
                label,
                repo_id or "(no repo_id — generate / build locally)",
            )

            validated_path: str | None = None

            try:
                resource_type = item["resource"]

                if resource_type == "gazetteers":
                    validated_path = self.downloader.check_gazetteer(local_path)

                elif resource_type in ("ner", "nel"):
                    assert repo_id
                    validated_path = self.downloader.download_hf(local_path, repo_id)

                elif resource_type == "vectorized_dbs":
                    assert item["task"]
                    gaz_path = self.resolver.get_gaz_path(item["lang"], item["task"])
                    nel_path, _ = self.resolver.get_nel_path(item["lang"])
                    validated_path = self.downloader.build_vector_db(
                        gaz_path, nel_path, local_path
                    )

            except Exception:
                logger.exception("Failed to process [%s] — skipping.", label)
                errors.append(label)
                continue

            if validated_path:
                self.update_registry(item["registry_keys"], validated_path)
                logger.info("[%s]  ✓ ready at %s", label, validated_path)
            else:
                logger.warning(
                    "[%s]  handler returned no path — registry not updated.", label
                )

        if errors:
            logger.error(
                "sanitize() finished with %d error(s): %s",
                len(errors),
                ", ".join(errors),
            )
        else:
            logger.info("sanitize() complete — all resources ready.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _label(item: PendingResource) -> str:
        parts = [item["resource"], item["lang"]]
        if item["task"]:
            parts.append(item["task"])
        return "/".join(parts)