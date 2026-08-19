"""
Pydantic models for the DT4H Common Data Model (CDM).

Two families of models live here:

*Output models* (``RecordMetadata``, ``Annotation``, ``NlpOutput``,
``NlpServiceInfo``, ``NlpResponse``) describe what the pipeline emits.  They are
permissive: almost every field is optional so that partial pipelines (NER-only
runs, callers that supply a sparse footer) still serialise successfully.

*Input models* (``RecordMetadataInput``, ``NlpInputDocument``) describe what an
input document must provide before it will be processed.  They are strict: the
seven fields the CDM marks as mandatory are required, and a file missing any of
them is rejected rather than silently annotated with nulls.

Controlled vocabularies
-----------------------
The CDM documents a closed set of values for several fields.  Those sets are
declared below as tuples and enforced with a *warning*, not an error: an
unexpected value is passed through verbatim and reported on the logger.  This is
deliberate — the NER models and the source records are produced by other teams,
so an unrecognised value is a signal worth surfacing but not a reason to discard
a document.
"""

import logging
from datetime import datetime
from typing import Annotated, Optional

from pydantic import AfterValidator, BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

ADMISSION_TYPES = ("inpatient", "ambulatory", "observation-encounter", "emergency")
RECORD_FORMATS = ("txt", "PDF", "XML", "json", "docx")
CHARACTER_ENCODINGS = ("ASCII", "UTF-8", "UTF-16", "UTF-32", "No encoding", "Unknown")
REPORT_LANGUAGES = ("en", "nl", "es", "it", "cs", "ro", "sv", "ca")
YES_NO = ("yes", "no")
CONCEPT_CLASSES = (
    "symptom",
    "disorder/disease",
    "procedure",
    "medication",
    "cardiology entity",
    "other",
)
NER_COMPONENT_TYPES = ("dictionary lookup", "transformer", "other")
NEL_COMPONENT_TYPES = ("lexical similarity", "transformer", "other")
CONTROLLED_VOCAB_NAMESPACES = (
    "UMLS", "SNOMED CT", "ICD10", "MedDRA", "ICD9", "DT4H", "HPO", "LOINC",
    "ISO", "GeoNames", "MeSH", "ESCO", "ATC", "ICPC", "other", "none",
)
CONTROLLED_VOCAB_SOURCES = ("original", "machine translation", "manual translation")


def _controlled(field_label: str, allowed: tuple[str, ...]):
    """Build an optional-string annotation that warns on out-of-vocabulary values.

    The value is always returned unchanged; only the logger is touched.  Used
    instead of ``Literal`` so that a value the CDM does not document degrades to
    a warning rather than failing validation for the whole document.
    """

    def _check(value: Optional[str]) -> Optional[str]:
        if value is not None and value not in allowed:
            logger.warning(
                "%s: value %r is not one of the CDM values (%s). Passed through unchanged.",
                field_label,
                value,
                " | ".join(allowed),
            )
        return value

    return Annotated[Optional[str], AfterValidator(_check)]


def _controlled_required(field_label: str, allowed: tuple[str, ...]):
    """As :func:`_controlled`, but the field is mandatory and may not be null."""

    def _check(value: str) -> str:
        if value not in allowed:
            logger.warning(
                "%s: value %r is not one of the CDM values (%s). Passed through unchanged.",
                field_label,
                value,
                " | ".join(allowed),
            )
        return value

    return Annotated[str, AfterValidator(_check)]


AdmissionType = _controlled("admission_type", ADMISSION_TYPES)
RecordFormat = _controlled("record_format", RECORD_FORMATS)
CharacterEncoding = _controlled("record_character_encoding", CHARACTER_ENCODINGS)
ReportLanguage = _controlled("report_language", REPORT_LANGUAGES)
Deidentified = _controlled("deidentified", YES_NO)
DeidentifiedRequired = _controlled_required("deidentified", YES_NO)
Negation = _controlled("negation", YES_NO)
ConceptClass = _controlled("concept_class", CONCEPT_CLASSES)
NerComponentType = _controlled("ner_component_type", NER_COMPONENT_TYPES)
NelComponentType = _controlled("nel_component_type", NEL_COMPONENT_TYPES)
ControlledVocabNamespace = _controlled("controlled_vocabulary_namespace", CONTROLLED_VOCAB_NAMESPACES)
ControlledVocabSource = _controlled("controlled_vocabulary_source", CONTROLLED_VOCAB_SOURCES)


def _now_iso() -> str:
    """Current local time as ``yyyy-MM-ddTHH:mm:ss.SSS+HH:MM``.

    Evaluated per call.  The previous implementation used ``datetime.now()`` as
    a plain default, which Python evaluates once at import time — every record
    in a run shared a single timestamp frozen at process start.
    """
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class RecordMetadata(BaseModel):
    """Record-level metadata as emitted by the pipeline.

    Deliberately permissive: the mandatory-field contract is enforced on input
    (see :class:`RecordMetadataInput`), not here, so that callers supplying a
    sparse footer over HTTP still receive a well-formed response.
    """

    # From the source record / footer
    clinical_site_id:                   Optional[str] = None
    patient_id:                         Optional[str] = None
    admission_id:                       Optional[str] = None
    admission_date:                     Optional[str] = None
    admission_type:                     AdmissionType = None
    record_id:                          Optional[str | int] = None
    record_type:                        Optional[str] = None
    record_format:                      RecordFormat = None
    record_creation_date:               Optional[str] = None
    record_lastupdate_date:             Optional[str] = None
    record_character_encoding:          CharacterEncoding = None
    record_extraction_date:             Optional[str] = None
    report_section:                     Optional[str] = None
    report_language:                    ReportLanguage = None
    deidentified:                       Deidentified = None
    deidentification_pipeline_name:     Optional[str] = None
    deidentification_pipeline_version:  Optional[str] = None

    # Set at inference time
    text:                               str
    nlp_processing_date:                str = Field(default_factory=_now_iso)
    nlp_processing_pipeline_name:       str
    nlp_processing_pipeline_version:    str = "1.0"


class Annotation(BaseModel):
    """A single detected entity, in CDM field names.

    ``concept_confidence`` carries the *NER* extraction confidence.  The CDM has
    a single confidence slot per annotation, so when a NEL stage is added its
    score has nowhere of its own to go — see ``docs/cdm_open_questions.md``.
    """

    concept_class:                              ConceptClass = None
    start_offset:                               Optional[int] = None
    end_offset:                                 Optional[int] = None
    concept_mention_string:                     Optional[str] = None
    concept_confidence:                         Optional[float] = None
    ner_component_type:                         NerComponentType = None
    ner_component_version:                      Optional[str] = None
    negation:                                   Negation = None
    negation_confidence:                        Optional[float] = None
    qualifier_negation:                         Optional[str] = None
    qualifier_temporal:                         Optional[str] = None
    dt4h_concept_identifier:                    Optional[str] = None
    nel_component_type:                         NelComponentType = None
    nel_component_version:                      Optional[str] = None
    controlled_vocabulary_namespace:            ControlledVocabNamespace = None
    controlled_vocabulary_version:              Optional[str] = None
    controlled_vocabulary_concept_identifier:   Optional[str] = None
    controlled_vocabulary_concept_official_term:Optional[str] = None
    controlled_vocabulary_source:               ControlledVocabSource = None
    symptom_date:                               Optional[str] = None


class NlpOutput(BaseModel):
    record_metadata:    RecordMetadata
    annotations:        list[Annotation]
    processing_success: bool = True


class NlpServiceInfo(BaseModel):
    service_app_name:   str = "DT4H NLP Processor"
    service_language:   str = "en"
    service_version:    str = "1.0"
    service_model:      str


class NlpResponse(BaseModel):
    nlp_output:         NlpOutput
    nlp_service_info:   NlpServiceInfo


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

#: Fields the CDM marks mandatory on an input record.  Required by
#: :class:`RecordMetadataInput`; a document missing any of them is rejected.
MANDATORY_INPUT_FIELDS = (
    "clinical_site_id",
    "patient_id",
    "admission_id",
    "record_id",
    "deidentified",
    "deidentification_pipeline_name",
    "deidentification_pipeline_version",
)


class RecordMetadataInput(BaseModel):
    """Record metadata as supplied by the caller in an input document.

    Differs from :class:`RecordMetadata` in three ways:

    * the seven fields in :data:`MANDATORY_INPUT_FIELDS` are required;
    * ``text`` defaults to the empty string — an empty value means "read the
      text from the sibling ``.txt`` file" rather than "invalid document";
    * the ``nlp_processing_*`` fields are absent, since they describe this
      pipeline's own run and are filled in on output.
    """

    # Mandatory
    clinical_site_id:                   str
    patient_id:                         str
    admission_id:                       str
    record_id:                          str | int
    deidentified:                       DeidentifiedRequired
    deidentification_pipeline_name:     str
    deidentification_pipeline_version:  str

    # Optional
    admission_date:                     Optional[str] = None
    admission_type:                     AdmissionType = None
    record_type:                        Optional[str] = None
    record_format:                      RecordFormat = None
    record_creation_date:               Optional[str] = None
    record_lastupdate_date:             Optional[str] = None
    record_character_encoding:          CharacterEncoding = None
    record_extraction_date:             Optional[str] = None
    report_section:                     Optional[str] = None
    report_language:                    ReportLanguage = None

    #: Empty means the text lives in the sibling ``.txt`` file.
    text:                               str = ""


class _NlpInputBody(BaseModel):
    record_metadata: RecordMetadataInput


class NlpInputDocument(BaseModel):
    """The subset of a CDM document that an input file is required to carry.

    Input files follow the same schema as the pipeline's output, so they may
    also contain ``annotations``, ``processing_success`` and ``nlp_service_info``.
    Those keys are ignored: this pipeline produces them.
    """

    nlp_output: _NlpInputBody
