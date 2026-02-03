'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Domain schemas and OOP hierarchy for medical documents (reports, prescriptions, labs, admission forms)."
'''

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

## -----------------------------
## Basic data structures (future NER ready)
## -----------------------------
@dataclass(frozen=True)
class NamedEntity:
    """
        A lightweight named entity structure (future NER integration)

        Attributes:
            text: Raw entity surface form
            label: Entity label (PERSON, ORG, DATE, etc.)
            start_char: Start offset in the source text
            end_char: End offset in the source text
            score: Optional confidence score
    """

    text: str
    label: str
    start_char: int
    end_char: int
    score: Optional[float] = None

@dataclass(frozen=True)
class PatientIdentity:
    """
        Patient identity information shared across most medical documents

        Attributes:
            first_name: Patient first name if available
            last_name: Patient last name if available
            full_name: Full name if available (fallback)
            birth_date: Patient birth date if available
            age_years: Patient age in years if available
            patient_id: Internal patient identifier if present
            sex: Patient sex if present
    """

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: Optional[str] = None
    birth_date: Optional[date] = None
    age_years: Optional[int] = None
    patient_id: Optional[str] = None
    sex: Optional[str] = None

@dataclass(frozen=True)
class DocumentSegment:
    """
        A continuous block of text extracted from a document

        Attributes:
            segment_id: Unique segment identifier
            text: Segment text content
            start_char: Start offset in the original document text
            end_char: End offset in the original document text
            meta: Additional metadata (section, page, etc.)
    """

    segment_id: str
    text: str
    start_char: int
    end_char: int
    meta: Dict[str, str] = field(default_factory=dict)

## -----------------------------
## Medical document base (common fields)
## -----------------------------
@dataclass
class MedicalDocumentBase:
    """
        Base class for medical documents

        This class anticipates future NER extraction by providing placeholders
        for patient identity and named entities

        Attributes:
            filename: Source file name
            raw_text: Full raw extracted text
            patient: Patient identity if extracted
            document_date: Document date if extracted
            allergies: Allergies if extracted
            segments: Segmented blocks for matching/classification
            entities: Named entities (future NER)
            meta: Additional document metadata
    """

    filename: str
    raw_text: str

    patient: Optional[PatientIdentity] = None
    document_date: Optional[date] = None
    allergies: List[str] = field(default_factory=list)

    segments: List[DocumentSegment] = field(default_factory=list)
    entities: List[NamedEntity] = field(default_factory=list)

    meta: Dict[str, str] = field(default_factory=dict)

    def add_segment(self, segment: DocumentSegment) -> None:
        """
            Add a segment to this document

            Args:
                segment: Segment to append
        """
        
        ## Keep ordering as added (typically textual order)
        self.segments.append(segment)

    def add_entity(self, entity: NamedEntity) -> None:
        """
            Add a named entity to this document

            Args:
                entity: Entity to append
        """
        
        ## Future NER will populate these
        self.entities.append(entity)

## -----------------------------
## Report documents (CRH / CRO / CRA shared fields)
## -----------------------------
@dataclass
class ClinicalReportBase(MedicalDocumentBase):
    """
        Base class for clinical reports with shared report-like fields

        CRH, CRO, CRA have overlapping structure (identity, dates, practitioner
        facility, conclusion). This base class captures that overlap

        Attributes:
            practitioner_name: Report author or responsible clinician
            facility_name: Facility/hospital/clinic name
            reason_for_visit: Admission reason or indication
            diagnosis: Diagnosis summary if extracted
            conclusion: Conclusion or discharge summary if extracted
    """

    practitioner_name: Optional[str] = None
    facility_name: Optional[str] = None
    reason_for_visit: Optional[str] = None
    diagnosis: Optional[str] = None
    conclusion: Optional[str] = None

@dataclass
class CRHDocument(ClinicalReportBase):
    """
        CRH document (Compte Rendu d'Hospitalisation)

        Notes:
            - Often includes a summary of the entire care episode
            - May embed other document types (prescriptions, labs, imaging)
    """

@dataclass
class CRODocument(ClinicalReportBase):
    """
    CRO document (Compte Rendu Operatoire)

    Notes:
        - Typically includes procedure steps and outcomes
    """

@dataclass
class CRADocument(ClinicalReportBase):
    """
        CRA document (Compte Rendu d'Anesthesie)

        Notes:
            - Typically includes anesthesia protocol and monitoring details
    """
    
## -----------------------------
## Prescriptions (meds / exams)
## -----------------------------
@dataclass
class PrescriptionDocument(MedicalDocumentBase):
    """
        Prescription document

        Attributes:
            prescription_type: 'medicaments' or 'examen' (or other)
            items: Parsed list of medications or requested exams (optional)
            instructions: General instructions if extracted
    """

    prescription_type: str = "unknown"
    items: List[str] = field(default_factory=list)
    instructions: Optional[str] = None

## -----------------------------
## Lab results
## -----------------------------
@dataclass
class LabDocument(MedicalDocumentBase):
    """
        Lab analysis document

        Attributes:
            lab_name: Lab name if extracted
            results: Key-value results (optional extraction later)
    """

    lab_name: Optional[str] = None
    results: Dict[str, str] = field(default_factory=dict)

## -----------------------------
## Admission / admin forms
## -----------------------------
@dataclass
class AdmissionFormDocument(MedicalDocumentBase):
    """
        Patient admission / administrative form document

        Attributes:
            address: Patient address if extracted
            insurance: Insurance/mutuelle if extracted
            contact_phone: Phone if extracted
    """

    address: Optional[str] = None
    insurance: Optional[str] = None
    contact_phone: Optional[str] = None

## -----------------------------
## Factory helper
## -----------------------------
def create_document_by_label(
    filename: str,
    raw_text: str,
    label: str,
) -> MedicalDocumentBase:
    """
        Create an appropriate document instance based on a label name

        Args:
            filename: Document filename
            raw_text: Raw text content
            label: Target label (crh, cro, cra, etc.)

        Returns:
            Instantiated document object
    """
    
    ## Keep mapping explicit for clarity
    if label == "crh":
        return CRHDocument(filename=filename, raw_text=raw_text)

    if label == "cro":
        return CRODocument(filename=filename, raw_text=raw_text)

    if label == "cra":
        return CRADocument(filename=filename, raw_text=raw_text)

    if label in {"ordonnance_medicaments", "ordonnance_examen"}:
        presc_type = "medicaments" if label == "ordonnance_medicaments" else "examen"
        return PrescriptionDocument(
            filename=filename,
            raw_text=raw_text,
            prescription_type=presc_type,
        )

    if label == "analyse_labo":
        return LabDocument(filename=filename, raw_text=raw_text)

    if label == "fiche_patient_admission":
        return AdmissionFormDocument(filename=filename, raw_text=raw_text)

    ## Default generic doc
    return MedicalDocumentBase(filename=filename, raw_text=raw_text)