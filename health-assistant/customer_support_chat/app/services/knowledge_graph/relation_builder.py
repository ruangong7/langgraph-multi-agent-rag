"""
Relation Builder — Constructs typed relationships between extracted entities.

Supports both LLM-based and rule-based relation extraction, producing a structured
knowledge graph that can be stored in NetworkX or Neo4j.
"""

from typing import List, Tuple, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.assistants.assistant_base import llm
from customer_support_chat.app.core.logger import logger
from .entity_extractor import ExtractedEntity


# ── Relation Type Definitions ────────────────────────────────────────────

RELATION_TYPES = {
    "FLIES_ON": "PERSON → FLIGHT_NUMBER (passenger booked on flight)",
    "DEPARTS_FROM": "FLIGHT_NUMBER → LOCATION (flight origin)",
    "ARRIVES_AT": "FLIGHT_NUMBER → LOCATION (flight destination)",
    "LOCATED_IN": "HOTEL_NAME → LOCATION (hotel location)",
    "OPERATED_BY": "FLIGHT_NUMBER → AIRLINE (flight operator)",
    "BOOKED_AT": "PERSON → HOTEL_NAME (customer hotel booking)",
    "RENTS": "PERSON → CAR_TYPE (customer car rental)",
    "ON_DATE": "FLIGHT_NUMBER → DATE (flight date)",
    "HAS_ORDER": "PERSON → ORDER_ID (customer order)",
    "CONTAINS": "ORDER_ID → PRODUCT_NAME (order line item)",
    "PARTICIPATES_IN": "PERSON → EXCURSION_NAME (trip participation)",
    "OWNS_POLICY": "POLICY_TERM → AIRLINE (policy ownership)",
    "HAS_PRICE": "PRODUCT_NAME → AMOUNT (product pricing)",
    "CONTACT_VIA": "PERSON → EMAIL|PHONE (contact method)",
    "RELATED_TO": "default fallback relation",
}

RELATION_HIERARCHY = {
    "spatial": ["LOCATED_IN", "DEPARTS_FROM", "ARRIVES_AT"],
    "temporal": ["ON_DATE"],
    "ownership": ["HAS_ORDER", "BOOKED_AT", "RENTS", "PARTICIPATES_IN"],
    "operational": ["OPERATED_BY", "OWNS_POLICY"],
    "composition": ["CONTAINS", "HAS_PRICE"],
    "contact": ["CONTACT_VIA"],
    "fallback": ["RELATED_TO"],
}


class EntityRelation(BaseModel):
    """A typed relationship between two extracted entities."""
    subject: ExtractedEntity = Field(description="Source entity (head)")
    relation_type: str = Field(description="Relation type from RELATION_TYPES")
    object: ExtractedEntity = Field(description="Target entity (tail)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: str = Field(default="", description="Evidence snippet from source text")
    bidirectional: bool = Field(default=False, description="Whether the relation is symmetric")
    metadata: Dict[str, str] = Field(default_factory=dict)

    def as_triple(self) -> Tuple[str, str, str]:
        """Return (subject_value, relation_type, object_value) triple."""
        return (self.subject.value, self.relation_type, self.object.value)

    def reverse(self) -> "EntityRelation":
        """Create the inverse relation."""
        inverse_map = {
            "FLIES_ON": "HAS_PASSENGER",
            "DEPARTS_FROM": "ORIGIN_OF",
            "ARRIVES_AT": "DESTINATION_OF",
            "LOCATED_IN": "HOSTS",
            "OPERATED_BY": "OPERATES",
            "BOOKED_AT": "BOOKED_BY",
            "RENTS": "RENTED_BY",
            "ON_DATE": "DATE_OF",
            "HAS_ORDER": "ORDERED_BY",
            "CONTAINS": "CONTAINED_IN",
            "PARTICIPATES_IN": "HAS_PARTICIPANT",
            "OWNS_POLICY": "POLICY_OF",
            "HAS_PRICE": "PRICE_OF",
            "CONTACT_VIA": "CONTACT_FOR",
        }
        return EntityRelation(
            subject=self.object,
            relation_type=inverse_map.get(self.relation_type, f"INV_{self.relation_type}"),
            object=self.subject,
            confidence=self.confidence,
            evidence=self.evidence,
            metadata=self.metadata,
        )


class RelationExtractionOutput(BaseModel):
    """LLM structured output for relation extraction."""
    relations: List[EntityRelation] = Field(description="All extracted relations")


# ── LLM-Based Relation Extraction ───────────────────────────────────────

RELATION_EXTRACTION_SYSTEM = """You are a relation extraction engine for a travel & e-commerce knowledge graph.

Given a list of extracted entities and the original text, identify ALL relationships between entities.

Valid relation types:
{relation_types}

Rules:
- Every relation must connect two entities from the provided list
- Include an evidence snippet from the original text
- Set confidence=1.0 for explicit relations, 0.8 for implicit/inferred ones
- Do NOT invent entities — only use the provided list
- Make sure subject and object reference EXACT entity values from the input

Output a JSON with a "relations" array.
"""

RELATION_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RELATION_EXTRACTION_SYSTEM),
    ("human",
     "Original text:\n{text}\n\n"
     "Extracted entities:\n{entities}\n\n"
     "Extract all relationships between these entities."),
])


def _build_relation_chain():
    """Build the structured relation extraction chain."""
    types_desc = "\n".join(f"- {k}: {v}" for k, v in RELATION_TYPES.items())
    prompt = RELATION_EXTRACTION_PROMPT.partial(relation_types=types_desc)
    return prompt | llm.with_structured_output(RelationExtractionOutput)


_relation_chain = _build_relation_chain()


# ── Rule-Based Relation Extraction ──────────────────────────────────────

def _extract_by_rules(entities: List[ExtractedEntity]) -> List[EntityRelation]:
    """
    Rule-based relation extraction as a fallback when LLM is unavailable or
    for deterministic, high-precision relations.

    Uses entity type co-occurrence patterns and proximity heuristics.
    """
    relations: List[EntityRelation] = []
    entity_by_type: Dict[str, List[ExtractedEntity]] = {}

    for e in entities:
        entity_by_type.setdefault(e.entity_type, []).append(e)

    # Rule: PERSON + FLIGHT_NUMBER → FLIES_ON
    if "PERSON" in entity_by_type and "FLIGHT_NUMBER" in entity_by_type:
        for person in entity_by_type["PERSON"]:
            for flight in entity_by_type["FLIGHT_NUMBER"]:
                relations.append(EntityRelation(
                    subject=person, relation_type="FLIES_ON", object=flight,
                    confidence=0.9, evidence="Rule: PERSON+FLIGHT_NUMBER co-occurrence",
                ))

    # Rule: FLIGHT_NUMBER + LOCATION (first=origin, second=dest)
    if "FLIGHT_NUMBER" in entity_by_type and "LOCATION" in entity_by_type:
        for flight in entity_by_type["FLIGHT_NUMBER"]:
            locations = entity_by_type["LOCATION"]
            if len(locations) >= 1:
                relations.append(EntityRelation(
                    subject=flight, relation_type="DEPARTS_FROM", object=locations[0],
                    confidence=0.85, evidence="Rule: flight first location → origin",
                ))
            if len(locations) >= 2:
                relations.append(EntityRelation(
                    subject=flight, relation_type="ARRIVES_AT", object=locations[1],
                    confidence=0.85, evidence="Rule: flight second location → destination",
                ))

    # Rule: PERSON + HOTEL_NAME → BOOKED_AT
    if "PERSON" in entity_by_type and "HOTEL_NAME" in entity_by_type:
        for person in entity_by_type["PERSON"]:
            for hotel in entity_by_type["HOTEL_NAME"]:
                relations.append(EntityRelation(
                    subject=person, relation_type="BOOKED_AT", object=hotel,
                    confidence=0.85, evidence="Rule: PERSON+HOTEL_NAME co-occurrence",
                ))

    # Rule: ORDER_ID + PRODUCT_NAME → CONTAINS
    if "ORDER_ID" in entity_by_type and "PRODUCT_NAME" in entity_by_type:
        for order in entity_by_type["ORDER_ID"]:
            for product in entity_by_type["PRODUCT_NAME"]:
                relations.append(EntityRelation(
                    subject=order, relation_type="CONTAINS", object=product,
                    confidence=0.9, evidence="Rule: ORDER+PRODUCT co-occurrence",
                ))

    # Rule: PERSON + ORDER_ID → HAS_ORDER
    if "PERSON" in entity_by_type and "ORDER_ID" in entity_by_type:
        for person in entity_by_type["PERSON"]:
            for order in entity_by_type["ORDER_ID"]:
                relations.append(EntityRelation(
                    subject=person, relation_type="HAS_ORDER", object=order,
                    confidence=0.9, evidence="Rule: PERSON+ORDER co-occurrence",
                ))

    # Rule: HOTEL_NAME + LOCATION → LOCATED_IN
    if "HOTEL_NAME" in entity_by_type and "LOCATION" in entity_by_type:
        for hotel in entity_by_type["HOTEL_NAME"]:
            for loc in entity_by_type["LOCATION"]:
                relations.append(EntityRelation(
                    subject=hotel, relation_type="LOCATED_IN", object=loc,
                    confidence=0.8, evidence="Rule: HOTEL+LOCATION co-occurrence",
                ))

    logger.info(f"📏 Rule-based extraction generated {len(relations)} relations")
    return relations


# ── Relation Builder Class ───────────────────────────────────────────────

class RelationBuilder:
    """
    Builds typed relationships between extracted entities.

    Supports two modes:
    1. LLM-based: Uses structured LLM for high-quality relation extraction
    2. Rule-based: Uses deterministic co-occurrence patterns (fast, no API cost)

    The LLM mode is preferred for accuracy; rule-based is a fallback.
    """

    def __init__(self, use_llm: bool = True):
        self._chain = _relation_chain if use_llm else None
        self._use_llm = use_llm

    def build_relations(
        self,
        text: str,
        entities: List[ExtractedEntity],
    ) -> List[EntityRelation]:
        """
        Extract relationships from text using extracted entities.

        Args:
            text: Original source text.
            entities: Previously extracted entities.

        Returns:
            List of EntityRelation objects.
        """
        if not entities or len(entities) < 2:
            logger.debug("Need at least 2 entities to build relations")
            return []

        logger.info(f"🔗 Building relations from {len(entities)} entities")

        # Try LLM extraction first
        llm_relations: List[EntityRelation] = []
        if self._use_llm:
            try:
                entities_json = [
                    {"type": e.entity_type, "value": e.value, "raw": e.raw_text}
                    for e in entities
                ]
                result = self._chain.invoke({
                    "text": text,
                    "entities": entities_json,
                })
                llm_relations = result.relations
                logger.info(f"🤖 LLM extracted {len(llm_relations)} relations")
            except Exception as exc:
                logger.warning(f"⚠️ LLM relation extraction failed: {exc}, using rules")

        # Merge with rule-based if LLM found nothing
        rule_relations = _extract_by_rules(entities)

        # Deduplicate: merge by triple key, keep highest confidence
        merged: Dict[str, EntityRelation] = {}
        for rel in rule_relations + llm_relations:
            key = f"{rel.subject.value}|{rel.relation_type}|{rel.object.value}"
            if key not in merged or rel.confidence > merged[key].confidence:
                merged[key] = rel

        all_relations = list(merged.values())
        logger.info(f"✅ Built {len(all_relations)} unique relations "
                     f"(LLM: {len(llm_relations)}, Rules: {len(rule_relations)})")
        return all_relations

    def build_from_conversation(
        self,
        messages: List[Any],
        entities: List[ExtractedEntity],
    ) -> List[EntityRelation]:
        """
        Build relations from a conversation context.

        Args:
            messages: Conversation messages.
            entities: Entities extracted from the conversation.

        Returns:
            Relations discovered from the conversation context.
        """
        full_text = " ".join(
            msg.content if hasattr(msg, 'content') else str(msg)
            for msg in messages[-5:]
            if hasattr(msg, 'content') and isinstance(msg.content, str)
        )
        return self.build_relations(full_text, entities)


# Singleton instance
relation_builder = RelationBuilder(use_llm=True)
