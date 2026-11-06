"""Report a linked concept under more than one terminology.

The linker resolves a mention to a code in its gazetteer's own terminology —
SNOMED CT for disorders, symptoms and procedures, a UMLS CUI for medications.
This module adds the equivalent codes in a second terminology, so a single
mention can be reported as, say, SNOMED CT ``1178005`` *and* ICD-10 ``G80.1``.

Two pieces:

* ``Crosswalk`` — one source → target code table, loaded once per file.
* ``Terminology`` — what a linked code of one entity type *is* (namespace and
  term provenance) and the crosswalks its codes are looked up in.

A mapping never re-scores anything. The mapped code inherits the link's
confidence: the only uncertain step was choosing the source code, and the
crosswalk adds no uncertainty of its own.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Which terminologies apply to which entity type
# ---------------------------------------------------------------------------

#: Registry entity type → (gazetteer terminology, crosswalks it is mapped through).
#: Every crosswalk contributes its targets, in this order: a SNOMED CT code with
#: both an ICD-10 and an ICD-9 mapping is reported under both.
ENTITY_TERMINOLOGIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "disease":   ("SNOMED CT", ("icd10", "icd9")),
    "symptom":   ("SNOMED CT", ("icd10", "icd9")),
    "procedure": ("SNOMED CT", ("icd10pcs", "icd9")),
    "drug":      ("UMLS",      ("atc",)),
}

#: Terminology for an entity type absent from ``ENTITY_TERMINOLOGIES``.
DEFAULT_NAMESPACE = "SNOMED CT"

#: Crosswalk name → CDM ``controlled_vocabulary_namespace`` of its targets.
CROSSWALK_NAMESPACES: dict[str, str] = {
    "icd10":    "ICD10",
    "icd10pcs": "ICD10PCS",
    "icd9":     "ICD9",
    "atc":      "ATC",
}

#: Crosswalks read from the entity's own gazetteer rather than a registry file:
#: name → the ``SAB`` value whose ``SAB_code`` holds the target code.  The UMLS
#: medication gazetteer already lists each CUI's ATC codes as rows of its own.
GAZETTEER_CROSSWALKS: dict[str, str] = {
    "atc": "ATC",
}

#: Languages whose SNOMED CT terms come from an official edition.  Every other
#: language's SNOMED CT gazetteer was machine-translated.
ORIGINAL_SNOMED_LANGUAGES = frozenset({"es", "en"})

ORIGINAL = "original"
MACHINE_TRANSLATION = "machine translation"


def term_source(namespace: str, lang: str) -> str:
    """CDM ``controlled_vocabulary_source``: how the gazetteer's terms were obtained."""
    if namespace == "SNOMED CT" and lang not in ORIGINAL_SNOMED_LANGUAGES:
        return MACHINE_TRANSLATION
    return ORIGINAL


def registry_crosswalks(entity: str) -> list[str]:
    """Crosswalk names an entity type needs from the registry's ``crosswalks`` section."""
    _, names = ENTITY_TERMINOLOGIES.get(entity, (DEFAULT_NAMESPACE, ()))
    return [name for name in names if name not in GAZETTEER_CROSSWALKS]


# ---------------------------------------------------------------------------
# Crosswalk
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Crosswalk:
    """Source code → target codes in one terminology.

    Targets are stored already reduced by ``most_specific``, so a lookup is a
    plain dictionary read.
    """

    namespace: str
    targets: Mapping[str, tuple[str, ...]] = field(repr=False)

    def __getitem__(self, code: str) -> tuple[str, ...]:
        return self.targets.get(code, ())

    @classmethod
    def from_pairs(cls, namespace: str, pairs) -> Crosswalk:
        grouped: dict[str, set[str]] = {}
        for source, target in pairs:
            grouped.setdefault(source, set()).add(target)
        return cls(namespace, {s: most_specific(t) for s, t in grouped.items()})


def most_specific(codes) -> tuple[str, ...]:
    """Drop every code that is a prefix of another, i.e. a parent of it.

    ICD, ICD-10-PCS and ATC are all hierarchical by prefix, and the mappings
    list a parent beside its children (``G80``, ``G80.1``, ``G80.9``).  The
    parent adds nothing the child does not already say.
    """
    codes = set(codes)
    return tuple(sorted(
        code for code in codes
        if not any(other != code and other.startswith(code) for other in codes)
    ))


_cache: dict[tuple, Crosswalk] = {}


def load_parquet_crosswalk(path: str | Path, namespace: str) -> Crosswalk:
    """Read a Parquet table with ``source_code`` and ``target_code`` columns."""
    key = ("parquet", str(path), namespace)
    if key not in _cache:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Crosswalk not found: {path}")
        rows = pd.read_parquet(path, columns=["source_code", "target_code"]).dropna()
        # astype(str) for the same reason load_gazetteer reads with dtype=str:
        # a SNOMED code stored as an integer must still match the linker's string.
        pairs = zip(rows["source_code"].astype(str), rows["target_code"].astype(str))
        _cache[key] = Crosswalk.from_pairs(namespace, pairs)
        logger.info("Crosswalk %s: %d source codes mapped to %s",
                    path.name, len(_cache[key].targets), namespace)
    return _cache[key]


def load_gazetteer_crosswalk(gaz_path: str | Path, sab: str, namespace: str) -> Crosswalk:
    """Read the ``code`` → ``SAB_code`` rows of a gazetteer whose ``SAB`` is *sab*.

    A gazetteer without ``SAB``/``SAB_code`` columns yields an empty crosswalk:
    its codes are still linked and reported, just never mapped.
    """
    key = ("gazetteer", str(gaz_path), sab, namespace)
    if key not in _cache:
        frame = pd.read_csv(gaz_path, sep="\t", dtype=str)
        if {"SAB", "SAB_code"} - set(frame.columns):
            logger.warning("Gazetteer %s has no SAB/SAB_code columns — no %s codes will be added.",
                           Path(gaz_path).name, namespace)
            pairs = []
        else:
            rows = frame.loc[frame["SAB"] == sab, ["code", "SAB_code"]].dropna()
            pairs = zip(rows["code"].str.strip(), rows["SAB_code"].str.strip())
        _cache[key] = Crosswalk.from_pairs(namespace, pairs)
    return _cache[key]


def clear_cache() -> None:
    _cache.clear()


# ---------------------------------------------------------------------------
# Terminology
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Terminology:
    """The terminology one entity type's gazetteer codes belong to."""

    namespace: str
    source: str
    crosswalks: tuple[Crosswalk, ...] = ()

    def concept_codes(self, code: str, term: str | None) -> list[dict]:
        """Every code a link is reported under: the gazetteer's own, then its mappings.

        Mapped codes carry no term — the crosswalks have no descriptions — and
        are always ``original``: ICD and ATC codes are never translated.
        """
        codes = [{"code": code, "term": term, "namespace": self.namespace, "source": self.source}]
        for crosswalk in self.crosswalks:
            codes += [
                {"code": target, "term": None, "namespace": crosswalk.namespace, "source": ORIGINAL}
                for target in crosswalk[code]
            ]
        return codes


def build_terminology(resolver, lang: str, entity: str) -> Terminology:
    """Assemble the ``Terminology`` for one registry entity type in one language."""
    namespace, names = ENTITY_TERMINOLOGIES.get(entity, (DEFAULT_NAMESPACE, ()))
    crosswalks = tuple(
        load_gazetteer_crosswalk(resolver.get_gaz_path(lang, entity), GAZETTEER_CROSSWALKS[name],
                                 CROSSWALK_NAMESPACES[name])
        if name in GAZETTEER_CROSSWALKS
        else load_parquet_crosswalk(resolver.get_crosswalk_path(name), CROSSWALK_NAMESPACES[name])
        for name in names
    )
    return Terminology(namespace, term_source(namespace, lang), crosswalks)


def attach_concept_codes(
    results: list[list[list[dict]]],
    terminologies: list[Terminology],
) -> list[list[list[dict]]]:
    """Add ``codes`` to every linked annotation, in place.

    ``results`` is nested entity type → document → annotation, aligned with
    ``terminologies`` on the outer level exactly as ``biencoder_inference``
    aligns it with its linkers.  An annotation without a ``code`` was never
    linked and gains nothing.
    """
    if len(results) != len(terminologies):
        raise ValueError(
            f"Got {len(results)} entity-type result groups but {len(terminologies)} "
            "terminologies — they must correspond one to one."
        )
    for docs, terminology in zip(results, terminologies):
        for doc in docs:
            for ann in doc:
                if "code" in ann:
                    ann["codes"] = terminology.concept_codes(ann["code"], ann.get("term"))
    return results
