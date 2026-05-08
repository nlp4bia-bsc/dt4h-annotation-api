"""
DT4H Common Data Model (CDM v2) formatter.

Transforms the raw NLP pipeline output into the DT4H CDM v2 JSON structure,
validating every field against the Pydantic models defined in
``data_structures.py``.

Field mapping
-------------
The table below shows how raw pipeline fields map to CDM v2 fields:

+----------------------+------------------------------------+----------------------------------+
| Raw field            | CDM v2 field                       | Notes                            |
+======================+====================================+==================================+
| ``ner_class``        | ``concept_class``                  | Spanish labels mapped to English |
|                      |                                    | literals (see ``_NER_CLASS_MAP``)|
+----------------------+------------------------------------+----------------------------------+
| ``start``            | ``start_offset``                   |                                  |
+----------------------+------------------------------------+----------------------------------+
| ``end``              | ``end_offset``                     |                                  |
+----------------------+------------------------------------+----------------------------------+
| ``span``             | ``mention_string``                 |                                  |
+----------------------+------------------------------------+----------------------------------+
| ``ner_score``        | ``extraction_confidence``          |                                  |
+----------------------+------------------------------------+----------------------------------+
| ``term``             | ``concept_str``                    |                                  |
+----------------------+------------------------------------+----------------------------------+
| ``code``             | ``concept_code``                   |                                  |
+----------------------+------------------------------------+----------------------------------+
| ``nel_score``        | ``concept_confidence``             |                                  |
+----------------------+------------------------------------+----------------------------------+
| ``is_negated``       | ``negation``                       | bool → ``"yes"`` / ``"no"``      |
+----------------------+------------------------------------+----------------------------------+
| ``negation_score``   | ``negation_confidence``            |                                  |
+----------------------+------------------------------------+----------------------------------+
| ``is_uncertain``     | ``uncertainty``                    | bool → ``"yes"`` / ``"no"``      |
+----------------------+------------------------------------+----------------------------------+
| ``uncertainty_score``| ``uncertainty_confidence``         |                                  |
+----------------------+------------------------------------+----------------------------------+

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
# Internal constants
# ---------------------------------------------------------------------------

# Maps Spanish NER labels produced by the pipeline to the CDM concept_class
# literals.  Labels not present here are lower-cased and passed through as-is,
# which handles already-English labels such as "procedure" or "medication".
_NER_CLASS_MAP: dict[str, str] = {
    "ENFERMEDAD": "disorder/disease",
    "SINTOMA": "symptom",
    "PROCEDIMIENTO": "procedure",
    "MEDICAMENTO": "medication",
}


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
        """Convert a single raw annotation dict to CDM v2 field names.

        Parameters
        ----------
        ann:
            A single raw annotation dict.

        Returns
        -------
        dict
            The same data with CDM v2 field names.

        Raises
        ------
        ValueError
            If any expected key is absent from ``ann``.
        """
        try:
            raw_class = ann["ner_class"]
            concept_class = _NER_CLASS_MAP.get(raw_class, raw_class.lower())

            return {
                "concept_class":        concept_class,
                "start_offset":         ann["start"],
                "end_offset":           ann["end"],
                "mention_string":       ann["span"],
                "extraction_confidence":ann["ner_score"],
                "concept_str":          ann["term"],
                "concept_code":         ann["code"],
                "concept_confidence":   ann["nel_score"],
                "negation":             "yes" if ann["is_negated"] else "no",
                "negation_confidence":  ann["negation_score"],
                "uncertainty":          "yes" if ann["is_uncertain"] else "no",
                "uncertainty_confidence":ann["uncertainty_score"],
            }
        except KeyError as exc:
            raise ValueError(f"Missing expected annotation field: {exc}") from exc