'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Canonical data schema for Clinical NER: Record and Entity models with strict business validation."
'''

from __future__ import annotations

## Standard library imports
from dataclasses import asdict, dataclass, field
from typing import Any

## Core domain imports
from src.core.entities import (
    DictionarySource,
    EntityLabel,
    EntityProvenance,
    NegationStatus,
    ensure_unique_entity_ids,
    normalize_dictionary_source,
    normalize_label,
    validate_temporality,
)

## Centralized errors and logging
from src.core.errors import DataError
from src.utils.logging_utils import get_logger

## Generic utilities
from src.utils.utils import (
    ensure_str,
    ensure_str_or_none,
    is_valid_entity_id,
    is_valid_patient_id,
    is_valid_record_id,
    json_dumps,
    load_list_of_dicts_from_json,
    parse_date_to_iso,
)

## Module-level logger
logger = get_logger(name="clinical_ner.schema")


@dataclass(slots=True)
class Entity:
    """
        Canonical entity annotation used across the Clinical NER pipeline

        Attributes:
            id: Unique entity identifier inside a record
            text: Surface form extracted from the document
            start: Start character offset in the document text
            end: End character offset in the document text
            label: Canonical entity label
            concept_id: Ontology or dictionary identifier (e.g. MeSH ID)
            concept_name: Canonical concept name from dictionary
            dictionary: Dictionary or ontology source
            negation: Negation status of the entity mention
            temporality: Temporal status (label-dependent)
            confidence: Optional confidence score in [0, 1]
            source: Provenance of the entity annotation
            meta: Free metadata container
    """

    ## Core span fields
    id: str
    text: str
    start: int
    end: int
    label: EntityLabel

    ## Dictionary linkage
    concept_id: str | None = None
    concept_name: str | None = None
    dictionary: DictionarySource = DictionarySource.MESH

    ## Clinical modifiers
    negation: NegationStatus = NegationStatus.UNKNOWN
    temporality: str | None = None

    ## Provenance and scoring
    confidence: float | None = None
    source: EntityProvenance = EntityProvenance.MANUAL

    ## Free metadata
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
            Validate entity internal consistency

            Raises:
                DataError: If any validation rule is violated
        """

        ## Validate entity id format
        if not is_valid_entity_id(self.id):
            msg = f"Invalid entity id format: {self.id}"
            logger.error(msg)
            raise DataError(msg)

        ## Validate surface text
        if not ensure_str(self.text).strip():
            msg = f"Empty entity text for entity id: {self.id}"
            logger.error(msg)
            raise DataError(msg)

        ## Validate span types
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            msg = f"Non-integer span for entity id: {self.id}"
            logger.error(msg)
            raise DataError(msg)

        ## Validate span ordering
        if self.start < 0 or self.end <= self.start:
            msg = (
                f"Invalid span for entity id {self.id}: "
                f"start={self.start}, end={self.end}"
            )
            logger.error(msg)
            raise DataError(msg)

        ## Normalize temporality before validating
        if self.temporality is not None:
            self.temporality = ensure_str(self.temporality).strip().lower()

        ## Validate temporality against label constraints
        if not validate_temporality(self.label, self.temporality):
            msg = (
                f"Invalid temporality '{self.temporality}' "
                f"for label '{self.label.value}' (entity id {self.id})"
            )
            logger.error(msg)
            raise DataError(msg)

        ## Validate confidence range and type
        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)):
                msg = f"Non-numeric confidence for entity id {self.id}"
                logger.error(msg)
                raise DataError(msg)

            if not 0.0 <= float(self.confidence) <= 1.0:
                msg = f"Confidence out of range for entity id {self.id}"
                logger.error(msg)
                raise DataError(msg)

        ## Validate meta container type
        if not isinstance(self.meta, dict):
            msg = f"Invalid meta type for entity id {self.id} (expected dict)"
            logger.error(msg)
            raise DataError(msg)

        ## Validate concept linkage for auto entities
        if self.source in (EntityProvenance.MESH_AUTO, EntityProvenance.DICT_AUTO):
            if not ensure_str_or_none(self.concept_id):
                msg = f"Missing concept_id for auto entity id {self.id}"
                logger.error(msg)
                raise DataError(msg)

            if not ensure_str_or_none(self.concept_name):
                msg = f"Missing concept_name for auto entity id {self.id}"
                logger.error(msg)
                raise DataError(msg)

    def to_dict(self) -> dict[str, Any]:
        """
            Convert entity to a JSON-serializable dictionary

            Returns:
                Dictionary representation of the entity
        """
        ## Convert dataclass to dict
        payload = asdict(self)

        ## Replace enums by raw values
        payload["label"] = self.label.value
        payload["dictionary"] = self.dictionary.value
        payload["negation"] = self.negation.value
        payload["source"] = self.source.value

        return payload

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Entity":
        """
            Build an Entity instance from a raw dictionary

            Args:
                data: Raw entity dictionary

            Returns:
                Entity instance

            Raises:
                DataError: If the payload is inconsistent
        """

        ## Normalize label
        label = normalize_label(ensure_str(data.get("label")))

        ## Normalize dictionary source
        dictionary = normalize_dictionary_source(
            ensure_str(data.get("dictionary", DictionarySource.MESH.value))
        )

        ## Normalize negation
        neg_raw = ensure_str(data.get("negation", NegationStatus.UNKNOWN.value)).lower()
        negation = next(
            (v for v in NegationStatus if v.value == neg_raw),
            NegationStatus.UNKNOWN,
        )

        ## Normalize provenance
        src_raw = ensure_str(data.get("source", EntityProvenance.MANUAL.value)).lower()
        source = next(
            (v for v in EntityProvenance if v.value == src_raw),
            EntityProvenance.MANUAL,
        )

        ## Normalize confidence
        conf_raw = data.get("confidence", None)
        confidence = None
        if conf_raw is not None:
            try:
                confidence = float(conf_raw)
            except Exception as exc:
                msg = "Invalid confidence type (expected float-compatible)"
                logger.error(msg)
                raise DataError(msg) from exc

        ## Normalize meta
        meta_raw = data.get("meta") or {}
        if not isinstance(meta_raw, dict):
            msg = "Invalid meta type (expected dict)"
            logger.error(msg)
            raise DataError(msg)

        ent = Entity(
            id=ensure_str(data.get("id")),
            text=ensure_str(data.get("text")),
            start=int(data.get("start", -1)),
            end=int(data.get("end", -1)),
            label=label,
            concept_id=data.get("concept_id"),
            concept_name=data.get("concept_name"),
            dictionary=dictionary,
            negation=negation,
            temporality=data.get("temporality"),
            confidence=confidence,
            source=source,
            meta=meta_raw,
        )

        ## Validate entity
        ent.validate()

        return ent


@dataclass(slots=True)
class Record:
    """
        Canonical record container for a single clinical document

        Attributes:
            record_id: Unique record identifier
            patient_id: Patient identifier
            name_document: Document filename or logical name
            type_document: Document type (free text)
            text: Raw clinical text
            date_document: ISO date string (YYYY-MM-DD) or None
            entities: List of extracted/annotated entities
    """

    record_id: str
    patient_id: str
    name_document: str
    type_document: str
    text: str
    date_document: str | None = None
    entities: list[Entity] = field(default_factory=list)

    def validate(self, validate_spans: bool = True, max_entities: int = 300) -> None:
        """
            Validate record internal consistency

            Args:
                validate_spans: Whether to validate entity spans against record text
                max_entities: Safety limit for number of entities

            Raises:
                DataError: If any validation rule is violated
        """

        ## Validate record_id and patient_id formats
        if not is_valid_record_id(self.record_id):
            msg = f"Invalid record_id format: {self.record_id}"
            logger.error(msg)
            raise DataError(msg)

        if not is_valid_patient_id(self.patient_id):
            msg = f"Invalid patient_id format: {self.patient_id}"
            logger.error(msg)
            raise DataError(msg)

        ## Validate required strings
        if not ensure_str(self.name_document).strip():
            msg = f"Empty name_document for record_id: {self.record_id}"
            logger.error(msg)
            raise DataError(msg)

        if not ensure_str(self.type_document).strip():
            msg = f"Empty type_document for record_id: {self.record_id}"
            logger.error(msg)
            raise DataError(msg)

        if not ensure_str(self.text).strip():
            msg = f"Empty text for record_id: {self.record_id}"
            logger.error(msg)
            raise DataError(msg)

        ## Normalize and validate date if provided
        if self.date_document is not None:
            self.date_document = parse_date_to_iso(self.date_document)

        ## Safety cap
        if len(self.entities) > max_entities:
            msg = (
                f"Too many entities ({len(self.entities)}) for record_id {self.record_id} "
                f"(max {max_entities})"
            )
            logger.error(msg)
            raise DataError(msg)

        ## Validate each entity
        for ent in self.entities:
            ent.validate()

        ## Validate unique entity ids
        if not ensure_unique_entity_ids([e.id for e in self.entities]):
            msg = f"Duplicate entity ids for record_id: {self.record_id}"
            logger.error(msg)
            raise DataError(msg)

        ## Validate entity spans inside record text
        if validate_spans:
            text_len = len(self.text)
            for ent in self.entities:
                if ent.end > text_len:
                    msg = (
                        f"Entity span out of bounds for record_id {self.record_id} "
                        f"(entity id {ent.id}, end={ent.end}, text_len={text_len})"
                    )
                    logger.error(msg)
                    raise DataError(msg)

                span_text = self.text[ent.start:ent.end]
                if not span_text:
                    msg = (
                        f"Empty span slice for record_id {self.record_id} "
                        f"(entity id {ent.id}, start={ent.start}, end={ent.end})"
                    )
                    logger.error(msg)
                    raise DataError(msg)

    def to_dict(self) -> dict[str, Any]:
        """
            Convert record to a JSON-serializable dictionary

            Returns:
                Dictionary representation of the record
        """
        payload = asdict(self)
        payload["entities"] = [e.to_dict() for e in self.entities]
        return payload

    def to_csv_row(self) -> dict[str, Any]:
        """
            Convert record to a flat CSV row

            Returns:
                Flat CSV row with entities stored as JSON string
        """
        return {
            "text": self.text,
            "name_document": self.name_document,
            "type_document": self.type_document,
            "patient_id": self.patient_id,
            "record_id": self.record_id,
            "date_document": self.date_document,
            "entities": json_dumps([e.to_dict() for e in self.entities]),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Record":
        """
            Build a Record instance from a raw dictionary

            Args:
                data: Raw record dictionary

            Returns:
                Record instance

            Raises:
                DataError: If the payload is inconsistent
        """

        ## Read entities payload
        entities_raw = data.get("entities", [])
        entities_list: list[dict[str, Any]]

        ## Accept string JSON or list[dict]
        if isinstance(entities_raw, str):
            entities_list = load_list_of_dicts_from_json(entities_raw)
        elif isinstance(entities_raw, list):
            entities_list = entities_raw
        else:
            msg = "Invalid entities format (expected JSON string or list)"
            logger.error(msg)
            raise DataError(msg)

        entities = [Entity.from_dict(e) for e in entities_list]

        rec = Record(
            record_id=ensure_str(data.get("record_id")),
            patient_id=ensure_str(data.get("patient_id")),
            name_document=ensure_str(data.get("name_document")),
            type_document=ensure_str(data.get("type_document")),
            text=ensure_str(data.get("text")),
            date_document=data.get("date_document"),
            entities=entities,
        )

        ## Validate record
        rec.validate(validate_spans=True)

        return rec
