"""
Use this formatter when no project-specific CDM transformation is required.

Output schema
-------------
The returned dict has the following top-level structure::

    {
        "metadata": {
            "text": "<original text>",
            # ... all footer key/value pairs passed through verbatim
        },
        "annotations": [
            {
                "start":            int,
                "end":              int,
                "ner_score":        float,
                "span":             str,
                "ner_class":        str,
                "code":             str,
                "term":             str,
                "nel_score":        float,
                "is_negated":       bool,
                "negation_score":   float,
                "is_uncertain":     bool,
                "uncertainty_score":float,
            },
            ...
        ],
        "processing_success": true,
        "processing_date": str
    }

No field renaming, type coercion, or controlled-vocabulary validation is
applied.
"""

from datetime import datetime
from typing import Optional

from app.src.format.base import DataFormatter

class PassthroughFormatter(DataFormatter):
    """Return pipeline output as-is with minimal envelope structure.

    This is the lean default for projects that do not impose a specific CDM.
    The ``footer`` dict and ``annotations`` list are passed through verbatim;
    only a thin wrapper is added to make the response self-describing.
    """

    def serialize(self, text: str, annotations: list[dict], footer: Optional[dict]) -> dict:
        """Wrap raw pipeline output in a minimal response envelope.

        Parameters
        ----------
        text:
            The clinical text that was processed.
        annotations:
            Raw annotation dicts from the NLP pipeline (not renamed or
            coerced).
        footer:
            Metadata dict supplied by the caller, if any; passed through unchanged.

        Returns
        -------
        dict
            A JSON-serialisable dict with ``metadata``, ``annotations``, and
            ``processing_success`` keys.
        """
        metadata = self._build_metadata(text, footer)
        return {
            "metadata": metadata,
            "annotations": annotations,
            "processing_success": True,
            "processing_date": datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        }

    # _transform_annotations and _build_metadata are intentionally not
    # overridden: the base class no-op implementations are the correct
    # behaviour for a passthrough formatter.