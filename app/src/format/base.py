"""
Abstract base class for NER+NEL output formatters.

Overview
--------
A *formatter* is responsible for taking the raw outputs of the NLP pipeline
(plain text, a list of annotation dicts, and an optional metadata footer) and
returning a serialised dict that can be sent directly as a JSON API response.

The default behaviour (see ``PassthroughFormatter`` in ``passthrough.py``) is
to return the data unchanged, keeping the core repository free of any
project-specific transformation logic.

To add a new output format (e.g. for a new CDM version), subclass
``DataFormatter``, implement ``serialize``, and – if field renaming or type
coercion is needed – override ``_transform_annotations`` and/or
``_build_metadata``.

Extension guide
---------------
1. Create a new file, e.g. ``format/myproject.py``.
2. Subclass ``DataFormatter``::

       from app.src.format.base import DataFormatter

       class MyProjectFormatter(DataFormatter):
           ...

3. Implement the three interface methods described below.
4. Export the class from ``format/__init__.py`` and register it in the
   ``method2formatter`` mapping in your application factory.

Input contract
--------------
``serialize`` always receives three arguments:

text : str
    The original clinical text that was processed by the pipeline.

annotations : list[dict]
    One dict per detected entity.  The *raw* schema produced by the pipeline
    is::

        {
            "start":            int,     # character start offset (inclusive)
            "end":              int,     # character end offset (exclusive)
            "ner_score":        float,   # NER confidence [0, 1]
            "span":             str,     # surface form as it appears in text
            "ner_class":        str,     # NER label (e.g. "ENFERMEDAD")
            "code":             str,     # linked concept code
            "term":             str,     # official term for the linked concept
            "nel_score":        float,   # NEL confidence [0, 1]
            "is_negated":       bool,
            "negation_score":   float,
            "is_uncertain":     bool,
            "uncertainty_score":float,
        }

footer : dict
    Arbitrary key/value metadata supplied by the caller (patient ID, site ID,
    record dates, encoding, …).  Its exact schema is caller-defined.

Output contract
---------------
``serialize`` must return a plain ``dict`` that is JSON-serialisable.  The
precise shape is left to each concrete implementation.
"""

from abc import ABC, abstractmethod
from typing import Optional


class DataFormatter(ABC):
    """Abstract base class for NLP output formatters.

    Concrete subclasses must implement :meth:`serialize`.  Optionally they may
    override :meth:`_transform_annotations` and :meth:`_build_metadata` when
    the transformation logic is complex enough to benefit from being split into
    separate, testable steps.

    Thread safety
    ~~~~~~~~~~~~~
    Formatters are expected to be *stateless*: ``serialize`` must not mutate
    any instance attribute.  This allows a single formatter instance to be
    reused concurrently across requests.
    """

    @abstractmethod
    def serialize(self, text: str, annotations: list[dict], footer: Optional[dict]) -> dict:
        """Transform raw pipeline output into the target JSON structure.

        Parameters
        ----------
        text:
            The clinical text that was processed.
        annotations:
            List of raw annotation dicts as produced by the NLP pipeline.
            See module docstring for the full field listing.
        footer:
            Metadata dict supplied by the API caller alongside the text, if any.

        Returns
        -------
        dict
            A JSON-serialisable dict whose schema is defined by the concrete
            implementation.

        Raises
        ------
        ValueError
            Implementations should raise ``ValueError`` (with a descriptive
            message) when a required field is missing or a value cannot be
            coerced to the expected type.
        """

    def _transform_annotations(self, annotations: list[dict]) -> list[dict]:
        """Rename / coerce annotation fields before validation.

        Override this method when the pipeline's raw annotation schema does
        not match the target schema.  The default implementation returns the
        list unchanged.

        Parameters
        ----------
        annotations:
            Raw annotation dicts from the pipeline.

        Returns
        -------
        list[dict]
            Annotation dicts ready for further processing or validation.
        """
        return annotations

    def _build_metadata(self, text: str, footer: Optional[dict]) -> dict:
        """Construct the record-metadata payload from the footer dict.

        Override this method to select, rename, or enrich footer fields.  The
        default implementation returns the footer as-is with ``text`` merged
        in.

        Parameters
        ----------
        text:
            The processed clinical text.
        footer:
            Raw metadata dict from the caller, if any.

        Returns
        -------
        dict
            A dict that will be included as the metadata section of the
            serialised response.
        """
        return {"text": text, **footer} if footer else {"text": text}