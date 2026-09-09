"""
DT4H Common Data Model (CDM v2) formatter.

Transforms the raw NLP pipeline output into the DT4H CDM v2 JSON structure,
validating every field against the Pydantic models defined in
``data_structures.py``.

Field mapping
-------------
The table below shows how raw pipeline fields map to CDM fields:

+----------------------+---------------------------------------------------+----------------------------------+
| Raw field            | CDM field                                         | Notes                            |
+======================+===================================================+==================================+
| ``ner_class``        | ``concept_class``                                 | mapped, see "Concept class"      |
+----------------------+---------------------------------------------------+----------------------------------+
| ``start``            | ``start_offset``                                  |                                  |
+----------------------+---------------------------------------------------+----------------------------------+
| ``end``              | ``end_offset``                                    |                                  |
+----------------------+---------------------------------------------------+----------------------------------+
| ``span``             | ``concept_mention_string``                        |                                  |
+----------------------+---------------------------------------------------+----------------------------------+
| ``nel_score``        | ``concept_confidence``                            | see "Confidence" below           |
+----------------------+---------------------------------------------------+----------------------------------+
| ``code``             | ``controlled_vocabulary_concept_identifier``      | absent for NER-only pipelines    |
+----------------------+---------------------------------------------------+----------------------------------+
| ``term``             | ``controlled_vocabulary_concept_official_term``   | absent for NER-only pipelines    |
+----------------------+---------------------------------------------------+----------------------------------+
| ``is_negated``       | ``negation``                                      | bool → ``"yes"`` / ``"no"``      |
+----------------------+---------------------------------------------------+----------------------------------+
| ``negation_score``   | ``negation_confidence``                           |                                  |
+----------------------+---------------------------------------------------+----------------------------------+

Concept class
-------------
``ner_class`` is the NER checkpoint's own label, read from its ``id2label``.
The checkpoints emit upper-case English labels (``DISEASE``, ``PROCEDURE``),
the older Spanish ones emit Spanish labels (``ENFERMEDAD``), and the registry
names the same entity types in lower case (``drug``).  The CDM vocabulary is
none of those: it is lower case, uses ``disorder/disease`` for disease and
``medication`` for drug.

``_to_concept_class`` maps between them.  A label it does not recognise is
passed through unchanged, so ``ConceptClass``'s validator still logs the
out-of-vocabulary warning naming the offending value — an unknown label stays
visible instead of being silently flattened to ``other``.

Confidence
----------
The CDM defines a single confidence slot per annotation, ``concept_confidence``.
It carries the **NEL linking confidence** — how confident the pipeline is that
the mention maps to the emitted ``controlled_vocabulary_concept_identifier``.

``ner_score`` has no CDM field of its own and is not serialised; it survives in
``PassthroughFormatter`` output.  See ``docs/cdm_open_questions.md``.

NER-only pipelines (``run_nerl.py``) produce no ``nel_score``, so they emit
``concept_confidence: null``.  That is the honest value: no linking claim was
made, and the field must not be back-filled with an extraction score, which
measures something else entirely.

Not assessed vs. assessed-negative
----------------------------------
``negation`` and ``negation_confidence`` are emitted as ``null`` when the raw
annotation carries no negation keys at all — i.e. when the negation model was
not run.  Only a pipeline that actually assessed the entity emits ``"no"``.
Reporting ``"no"`` for an unassessed entity would state a clinical finding the
pipeline never made.

``is_uncertain`` / ``uncertainty_score`` are produced by the negation stage but
have no CDM field; they survive in ``PassthroughFormatter`` output only.

Footer fields are mapped directly onto ``RecordMetadata`` using its declared
field names; unknown footer keys are silently ignored.
"""

import re

from app.src.format.base import DataFormatter
from app.src.format.data_structures import (
    Annotation,
    NlpOutput,
    NlpResponse,
    NlpServiceInfo,
    RecordMetadata,
)

# ---------------------------------------------------------------------------
# Concept class mapping
# ---------------------------------------------------------------------------

#: Raw NER labels (normalised by :func:`_normalise_label`) → CDM values.
#: Covers the English checkpoint labels, the Spanish ones, and the registry's
#: own entity-type names, since all three reach this code.  Every value must be
#: a member of ``data_structures.CONCEPT_CLASSES``.
CONCEPT_CLASS_MAP: dict[str, str] = {
    # disorder/disease
    "disease":            "disorder/disease",
    "diseases":           "disorder/disease",
    "disorder":           "disorder/disease",
    "disorders":          "disorder/disease",
    "disorder/disease":   "disorder/disease",
    "enfermedad":         "disorder/disease",
    "enfermedades":       "disorder/disease",
    # symptom
    "symptom":            "symptom",
    "symptoms":           "symptom",
    "sintoma":            "symptom",
    "síntoma":            "symptom",
    "sintomas":           "symptom",
    "síntomas":           "symptom",
    # procedure
    "procedure":          "procedure",
    "procedures":         "procedure",
    "procedimiento":      "procedure",
    "procedimientos":     "procedure",
    # medication — the registry entity type is 'drug', the CDM value is not
    "medication":         "medication",
    "medications":        "medication",
    "drug":               "medication",
    "drugs":              "medication",
    "med":                "medication",
    "meds":               "medication",
    "medicamento":        "medication",
    "medicamentos":       "medication",
    "farmaco":            "medication",
    "fármaco":            "medication",
    # already-CDM values, so a conformant checkpoint round-trips untouched
    "cardiology entity":  "cardiology entity",
    "other":              "other",
}

#: BIO tagging prefix left on a label when the HF pipeline aggregates with
#: ``aggregation_strategy="none"``.
_BIO_PREFIX = re.compile(r"^[BIOES]-")


def _normalise_label(label: str) -> str:
    """Reduce a raw NER label to the form used as a ``CONCEPT_CLASS_MAP`` key.

    Strips any BIO prefix, lower-cases, and treats ``_`` and ``-`` as spaces so
    that ``CARDIOLOGY_ENTITY`` and ``cardiology entity`` are the same key.
    ``/`` is preserved — ``disorder/disease`` is a CDM value in its own right.
    """
    label = _BIO_PREFIX.sub("", label.strip())
    label = label.replace("_", " ").replace("-", " ").lower()
    return " ".join(label.split())


def _to_concept_class(ner_class):
    """Map a raw NER label onto the CDM ``concept_class`` vocabulary.

    Unrecognised labels are returned unchanged rather than coerced to
    ``other``: ``ConceptClass`` warns on them by name, which is the signal that
    a checkpoint emits something this table has not been told about.  Silently
    flattening them would destroy exactly that signal.

    Non-string values pass through untouched so that malformed pipeline output
    is reported by Pydantic, not swallowed here.
    """
    if not isinstance(ner_class, str):
        return ner_class
    return CONCEPT_CLASS_MAP.get(_normalise_label(ner_class), ner_class)


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class Dt4hFormatter(DataFormatter):
    """Serialize pipeline output into the DT4H CDM v2 JSON structure.

    All annotation fields are renamed and coerced via
    :meth:`_transform_annotations`; footer fields are filtered to only those
    recognised by :class:`~data_structures.RecordMetadata` via
    :meth:`_build_metadata`.  The final dict is produced by Pydantic's
    ``model_dump`` so every value is guaranteed to be JSON-serialisable and
    type-valid.
    """

    def serialize(self, text: str, annotations: list[dict], footer: dict) -> dict:
        """Return a DT4H CDM v2-compliant response dict.

        Parameters
        ----------
        text:
            The clinical text that was processed.
        annotations:
            Raw annotation dicts from the NLP pipeline.
        footer:
            Metadata dict supplied by the caller.  Only keys that are declared
            fields of :class:`~data_structures.RecordMetadata` are forwarded;
            all others are silently dropped.

        Returns
        -------
        dict
            A JSON-serialisable dict matching the ``NlpResponse`` schema.

        Raises
        ------
        ValueError
            If a required annotation field is absent from one of the dicts in
            ``annotations``.
        """
        transformed = self._transform_annotations(annotations)
        validated_annotations = [Annotation(**ann) for ann in transformed]

        metadata_payload = self._build_metadata(text, footer)
        record_metadata = RecordMetadata(**metadata_payload)

        response = NlpResponse(
            nlp_output=NlpOutput(
                record_metadata=record_metadata,
                annotations=validated_annotations,
            ),
            nlp_service_info=NlpServiceInfo(
                service_model=self.__class__.__name__,
            ),
        )
        return response.model_dump(mode="json")

    # ------------------------------------------------------------------
    # Overridden hooks
    # ------------------------------------------------------------------

    def _transform_annotations(self, annotations: list[dict]) -> list[dict]:
        """Rename and coerce raw pipeline annotation fields to CDM v2 names.

        Parameters
        ----------
        annotations:
            Raw annotation dicts as produced by the pipeline.

        Returns
        -------
        list[dict]
            Annotation dicts with CDM v2 field names, ready to be passed to
            ``Annotation(**ann)``.

        Raises
        ------
        ValueError
            If a required field is missing from any annotation dict.
        """
        return [self._rename_annotation(ann) for ann in annotations]

    def _build_metadata(self, text: str, footer: dict) -> dict:
        """Extract recognised ``RecordMetadata`` fields from the footer.

        Unknown footer keys are silently ignored so that callers can include
        extra tracking fields without causing validation errors.

        Parameters
        ----------
        text:
            The processed clinical text.
        footer:
            Raw metadata dict from the caller.

        Returns
        -------
        dict
            A dict containing only keys declared on ``RecordMetadata``, plus
            the mandatory ``text`` and pipeline-identity fields.
        """
        known_fields = RecordMetadata.model_fields
        filtered_footer = {k: footer[k] for k in known_fields if k in footer}
        return {
            **filtered_footer,
            "text": text,
            "nlp_processing_pipeline_name": self.__class__.__name__,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rename_annotation(ann: dict) -> dict:
        """Convert a single raw annotation dict to CDM field names.

        Only the four span fields are required.  Linking and negation fields are
        optional so that NER-only pipelines (see ``run_nerl.py``) serialise
        without having to stub them out; anything absent becomes ``null``.

        Parameters
        ----------
        ann:
            A single raw annotation dict.

        Returns
        -------
        dict
            The same data with CDM field names.

        Raises
        ------
        ValueError
            If one of the required span fields is absent from ``ann``.
        """
        try:
            renamed = {
                "concept_class":            _to_concept_class(ann["ner_class"]),
                "start_offset":             ann["start"],
                "end_offset":               ann["end"],
                "concept_mention_string":   ann["span"],
            }
        except KeyError as exc:
            raise ValueError(f"Missing expected annotation field: {exc}") from exc

        # --- Linking (absent for NER-only pipelines) ---
        # concept_confidence is the *linking* confidence; a run with no NEL
        # stage emits null rather than falling back to the extraction score.
        renamed["concept_confidence"] = ann.get("nel_score")
        renamed["controlled_vocabulary_concept_identifier"] = ann.get("code")
        renamed["controlled_vocabulary_concept_official_term"] = ann.get("term")

        # --- Negation (absent when the negation model was not run) ---
        if "is_negated" in ann:
            renamed["negation"] = "yes" if ann["is_negated"] else "no"
            renamed["negation_confidence"] = ann.get("negation_score")

        return renamed
