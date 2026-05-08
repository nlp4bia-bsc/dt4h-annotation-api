from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime

# --- Enums / Literals for constrained fields ---

AdmissionType = Literal["inpatient", "ambulatory", "observation-encounter", "emergency"]
RecordFormat = Literal["txt", "PDF", "XML", "json", "docx"]
CharacterEncoding = Literal["ASCII", "UTF-8", "UTF-16", "UTF-32", "No encoding", "Unknown"]
ReportLanguage = Literal["en", "nl", "es", "it", "cs", "ro", "sv", "ca"]
Deidentified = Literal["yes", "no"]
ConceptClass = Literal["symptom", "disorder/disease", "procedure", "medication"] # Literal["symptom", "disorder/disease", "procedure", "medication", "cardiology entity", "other"]
NerComponentType = Literal["dictionary lookup", "transformer", "other"]
NelComponentType = Literal["lexical similarity", "transformer", "other"]
ControlledVocabNamespace = Literal["UMLS", "SNOMED CT", "ICD10", "MedDRA", "ICD9", "DT4H", "HPO", "LOINC", "ISO", "GeoNames", "MeSH", "ESCO", "ATC", "ICPC", "other", "none"]
ControlledVocabSource = Literal["original", "machine translation", "manual translation"]


# --- Sub-models ---

class RecordMetadata(BaseModel):
    # From footer
    clinical_site_id:                   Optional[str] = None
    patient_id:                         Optional[str] = None
    admission_id:                       Optional[str] = None
    admission_date:                     Optional[str] = None
    admission_type:                     Optional[AdmissionType] = None
    record_id:                          Optional[str | int] = None
    record_type:                        Optional[str] = None
    record_format:                      Optional[RecordFormat] = None
    record_creation_date:               Optional[str] = None
    record_lastupdate_date:             Optional[str] = None
    record_character_encoding:          Optional[CharacterEncoding] = None
    record_extraction_date:             Optional[str] = None
    report_section:                     Optional[str] = None
    report_language:                    Optional[ReportLanguage] = None
    deidentified:                       Optional[Deidentified] = None
    deidentification_pipeline_name:     Optional[str] = None
    deidentification_pipeline_version:  Optional[str] = None

    # Set at inference time
    text:                               str
    nlp_processing_date:                str = datetime.now().isoformat()
    nlp_processing_pipeline_name:       str
    nlp_processing_pipeline_version:    str = "1.0"


class Annotation(BaseModel):
    concept_class:                              Optional[ConceptClass] = None
    start_offset:                               Optional[int] = None
    end_offset:                                 Optional[int] = None
    mention_string:                             Optional[str] = None
    extraction_confidence:                      Optional[float] = None
    concept_str:                                Optional[str] = None
    concept_code:                               Optional[str] = None
    concept_confidence:                         Optional[float] = None
    ner_component_type:                         Optional[NerComponentType] = None
    ner_component_version:                      Optional[str] = None
    negation:                                   Optional[Deidentified] = None  # "yes" | "no"
    negation_confidence:                        Optional[float] = None
    uncertainty:                                   Optional[Deidentified] = None  # "yes" | "no"
    uncertainty_confidence:                        Optional[float] = None
    # qualifier_negation:                         Optional[str] = None
    # qualifier_temporal:                         Optional[str] = None
    dt4h_concept_identifier:                    Optional[str] = None
    nel_component_type:                         Optional[NelComponentType] = None
    nel_component_version:                      Optional[str] = None
    controlled_vocabulary_namespace:            Optional[ControlledVocabNamespace] = None
    controlled_vocabulary_version:              Optional[str] = None
    controlled_vocabulary_concept_identifier:   Optional[str] = None
    controlled_vocabulary_concept_official_term:Optional[str] = None
    controlled_vocabulary_source:               Optional[ControlledVocabSource] = None
    # symptom_date:                               Optional[str] = None


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