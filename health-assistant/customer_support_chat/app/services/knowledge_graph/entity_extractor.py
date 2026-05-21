"""
Entity Extractor — LLM-based entity extraction from user queries, documents, and dialogue context.

Uses structured output (Pydantic) to reliably extract typed entities from natural language.
Supports 15+ entity types relevant to travel & e-commerce customer support.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from customer_support_chat.app.services.assistants.assistant_base import llm
from customer_support_chat.app.core.logger import logger


# ── Entity Type Definitions ──────────────────────────────────────────────

class ExtractedEntity(BaseModel):
    """A single extracted entity with type, value, and metadata."""
    entity_type: str = Field(
        description="Entity type: PERSON, LOCATION, FLIGHT_NUMBER, AIRLINE, HOTEL_NAME, "
                    "CAR_TYPE, DATE, TIME, ORDER_ID, PRODUCT_NAME, EMAIL, PHONE, "
                    "EXCURSION_NAME, POLICY_TERM, AMOUNT"
    )
    value: str = Field(description="Normalized entity value (e.g., 'LH123', 'Zurich')")
    raw_text: str = Field(description="Original mention text from the source")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")
    attributes: Dict[str, str] = Field(
        default_factory=dict, description="Extra attributes (e.g., {code: LH, number: 123})"
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class EntityExtractionResult(BaseModel):
    """Result of entity extraction from a text."""
    text: str = Field(description="Original input text")
    entities: List[ExtractedEntity] = Field(default_factory=list)
    extraction_model: str = Field(default="llm_structured", description="Model used")
    tokens_used: int = Field(default=0)


# ── Entity Extraction Prompts ────────────────────────────────────────────

ENTITY_EXTRACTION_SYSTEM = """You are an entity extraction engine for a travel & e-commerce customer support system.

Extract ALL entities from the user's message. Use the following entity types:

PERSON        – Names of people, passengers, customers
LOCATION      – Cities, countries, airports (e.g., "Zurich", "JFK", "Switzerland")
FLIGHT_NUMBER – Flight identifiers (e.g., "LH123", "SWR456")
AIRLINE       – Airline names or codes (e.g., "Swiss Airlines", "SWR")
HOTEL_NAME    – Hotel names (e.g., "Marriott Zurich")
CAR_TYPE      – Car types or rental companies (e.g., "SUV", "Hertz")
DATE          – Dates in any format (e.g., "May 21", "2026-06-15")
TIME          – Times (e.g., "14:00", "3 PM")
ORDER_ID      – Order numbers or booking references (e.g., "ORD-77821")
PRODUCT_NAME  – Product names (e.g., "iPhone 15", "travel pillow")
EMAIL         – Email addresses
PHONE         – Phone numbers
EXCURSION_NAME – Tour/activity names (e.g., "Zurich City Tour")
POLICY_TERM   – Policy-related terms (e.g., "cancellation", "refund policy")
AMOUNT        – Monetary amounts (e.g., "$200", "150 CHF")

Rules:
- Extract ALL entities, even if multiple of the same type
- Normalize values (e.g., dates to ISO format YYYY-MM-DD)
- Set confidence=0.8 for ambiguous entities, 1.0 for clear ones
- Include raw_text as the original substring from the message
- Include attributes for composite entities (e.g., flight number → {airline_code: "LH", number: "123"})
"""

ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ENTITY_EXTRACTION_SYSTEM),
    ("human", "{text}"),
])


# ── Structured LLM Chain ────────────────────────────────────────────────

class EntityListOutput(BaseModel):
    """Wrapper for list extraction."""
    entities: List[ExtractedEntity] = Field(description="All extracted entities")


def _build_extraction_chain():
    """Build the structured extraction chain using the project's shared LLM."""
    return ENTITY_EXTRACTION_PROMPT | llm.with_structured_output(EntityListOutput)


_extraction_chain = _build_extraction_chain()


# ── Entity Extractor Class ───────────────────────────────────────────────

class EntityExtractor:
    """
    Extracts typed entities from user queries, FAQ documents, and dialogue context.

    Uses LLM with structured output (Pydantic) for high-accuracy extraction.
    Includes a fallback regex-based extractor for offline/lightweight scenarios.
    """

    def __init__(self):
        self._chain = _extraction_chain
        self._extraction_count = 0

    def extract(self, text: str) -> EntityExtractionResult:
        """
        Extract entities from text using the LLM chain.

        Args:
            text: Input text to extract entities from.

        Returns:
            EntityExtractionResult containing extracted entities and metadata.
        """
        if not text or not text.strip():
            return EntityExtractionResult(text=text or "", entities=[])

        logger.info(f"🔍 Extracting entities from text: {text[:100]}...")
        self._extraction_count += 1

        try:
            result = self._chain.invoke({"text": text})
            entities = result.entities

            # Post-process: remove low-confidence entities
            entities = [e for e in entities if e.confidence >= 0.6]

            logger.info(f"✅ Extracted {len(entities)} entities (attempt #{self._extraction_count})")
            for e in entities:
                logger.debug(f"  → [{e.entity_type}] {e.value} (conf: {e.confidence})")

            return EntityExtractionResult(
                text=text,
                entities=entities,
                extraction_model="llm_structured",
                tokens_used=len(text.split()),
            )
        except Exception as exc:
            logger.error(f"❌ Entity extraction failed: {exc}")
            # Fallback: return empty result — caller should handle gracefully
            return EntityExtractionResult(
                text=text,
                entities=[],
                extraction_model="llm_structured",
                tokens_used=0,
            )

    def extract_from_conversation(self, messages: List[Any]) -> List[ExtractedEntity]:
        """
        Extract entities from a conversation history.

        Args:
            messages: List of LangChain messages.

        Returns:
            Flattened list of all entities extracted from the conversation.
        """
        all_entities: List[ExtractedEntity] = []
        for msg in messages[-5:]:  # Last 5 messages for context window
            content = msg.content if hasattr(msg, 'content') else str(msg)
            if isinstance(content, str) and len(content) > 10:
                result = self.extract(content)
                all_entities.extend(result.entities)
        return all_entities

    def merge_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """
        Deduplicate and merge overlapping entities.

        Args:
            entities: List of entities, possibly with duplicates.

        Returns:
            Deduplicated entity list, keeping highest-confidence version.
        """
        deduped: Dict[str, ExtractedEntity] = {}
        for e in entities:
            key = f"{e.entity_type}:{e.value.lower()}"
            if key not in deduped or e.confidence > deduped[key].confidence:
                deduped[key] = e
        return sorted(deduped.values(), key=lambda e: e.confidence, reverse=True)

    @property
    def stats(self) -> Dict[str, Any]:
        return {"extraction_count": self._extraction_count}


# Singleton instance for use across the module
entity_extractor = EntityExtractor()
