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
| ``ner_class``        | ``concept_class``                                 | passed through verbatim          |
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

Confidence
----------
The CDM defines a single confidence slot per annotation, ``concept_confidence``.
It carries the **NEL linking confidence** — how confident the pipeline is that
the mention maps to the emitted ``controlled_vocabulary_concept_identifier``.

``ner_score`` has no CDM field of its own and is not serialised; it survives in
``PassthroughFormatter`` output.  See ``docs/cdm_open_questions.md``.

NER-only pipelines (``run_ner.py``) produce no ``nel_score``, so they emit
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

from app.src.format.base import DataFormatter
from app.src.format.data_structures import (
    Annotation,
    NlpOutput,
    NlpResponse,
    NlpServiceInfo,
    RecordMetadata,
)

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
        optional so that NER-only pipelines (see ``run_ner.py``) serialise
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
                "concept_class":            ann["ner_class"],
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
