"""
Medical Knowledge Graph Agent — Disease-Symptom-Medication-Doctor entity extraction
and multi-hop reasoning for the Health Assistant.

Integrates with the LangGraph StateGraph as a sub-agent node.
"""

from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, AIMessage
from langchain_openai import ChatOpenAI

from customer_support_chat.app.core.settings import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, KG_STORE_PATH
from customer_support_chat.app.core.logger import logger
from customer_support_chat.app.services.knowledge_graph.entity_extractor import (
    EntityExtractor, ExtractedEntity,
)
from customer_support_chat.app.services.knowledge_graph.relation_builder import (
    RelationBuilder, EntityRelation,
)
from customer_support_chat.app.services.knowledge_graph.graph_store import GraphStore
from customer_support_chat.app.services.knowledge_graph.reasoning import (
    KnowledgeReasoner, ReasoningResult,
)
from customer_support_chat.app.services.knowledge_graph.kg_retriever import (
    KGRetriever, RetrievalContext,
)


# ── Medical-specific entity and relation type mappings ────────────────

MEDICAL_ENTITY_TYPES = [
    "DISEASE", "SYMPTOM", "MEDICATION", "DOSAGE", "DOCTOR",
    "HOSPITAL", "DEPARTMENT", "PATIENT", "LAB_TEST", "VITAL_SIGN",
    "ALLERGEN", "PROCEDURE", "BODY_PART", "DATE", "LIFESTYLE_FACTOR",
]

MEDICAL_RELATION_TYPES = [
    "HAS_SYMPTOM", "TREATS", "TAKES", "PRESCRIBED_BY",
    "WORKS_AT", "LOCATED_IN", "DIAGNOSED_WITH", "ALLERGIC_TO",
    "HAS_RESULT", "MEASURED_AS", "LEADS_TO", "CONTRAINDICATED_WITH",
    "SCHEDULED_ON", "BELONGS_TO",
]


# ── LLM setup ─────────────────────────────────────────────────────────

medical_llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    temperature=0.1,
)


# ── Medical KG components ─────────────────────────────────────────────

entity_extractor = EntityExtractor(llm=medical_llm, entity_types=MEDICAL_ENTITY_TYPES)
relation_builder = RelationBuilder(llm=medical_llm, relation_types=MEDICAL_RELATION_TYPES)
graph_store = GraphStore(store_path=KG_STORE_PATH)
reasoner = KnowledgeReasoner(graph_store=graph_store)
kg_retriever = KGRetriever(graph_store=graph_store)


# ── KG Agent Tools ────────────────────────────────────────────────────

@tool
def extract_medical_entities(text: str) -> Dict[str, Any]:
    """
    Extract medical entities (diseases, symptoms, medications, doctors, etc.) from text.
    Returns a list of extracted entities with types and confidence scores.
    """
    try:
        entities = entity_extractor.extract(text)
        result = {
            "entity_count": len(entities),
            "entities": [
                {"type": e.entity_type, "value": e.value, "confidence": e.confidence}
                for e in entities
            ],
        }
        logger.info(f"🧬 Extracted {len(entities)} medical entities from text")
        return result
    except Exception as e:
        logger.error(f"Entity extraction error: {e}")
        return {"error": str(e), "entities": []}


@tool
def build_medical_relations(text: str) -> Dict[str, Any]:
    """
    Build medical knowledge relations from text (e.g., disease-has_symptom-fever).
    Extracts entities first, then identifies relationships between them.
    """
    try:
        entities = entity_extractor.extract(text)
        relations = relation_builder.extract_relations(text, entities)
        # Store in graph
        for e in entities:
            graph_store.add_entity(e)
        for r in relations:
            graph_store.add_relation(r)
        result = {
            "relation_count": len(relations),
            "relations": [
                {"subject": r.subject.value, "relation": r.relation_type, "object": r.object.value, "confidence": r.confidence}
                for r in relations
            ],
        }
        logger.info(f"🔗 Built {len(relations)} medical relations")
        return result
    except Exception as e:
        logger.error(f"Relation building error: {e}")
        return {"error": str(e), "relations": []}


@tool
def query_medical_kg(query: str) -> Dict[str, Any]:
    """
    Query the medical knowledge graph to find related entities and paths.
    Use for questions like 'What medications treat hypertension?' or
    'What are the side effects of metformin?'
    """
    try:
        entities = entity_extractor.extract(query)
        if not entities:
            return {"error": "No medical entities found in query", "results": []}
        
        results = []
        for entity in entities[:3]:  # Max 3 entities
            entity_id = graph_store.add_entity(entity)
            neighbors = graph_store.get_neighbors(entity_id, hops=2)
            results.append({
                "entity": entity.value,
                "type": entity.entity_type,
                "neighbors": [
                    {"value": n["node"].value if n["node"] else "unknown",
                     "type": n["node"].entity_type if n["node"] else "unknown",
                     "relations": [r["type"] for r in n["relations"]]}
                    for n in neighbors.get("neighbors", [])[:10]
                ],
            })
        
        # Also try reasoning for multi-hop paths
        found_entities = graph_store.find_entities(value_contains=entities[0].value)
        if len(found_entities) >= 2:
            paths = reasoner.find_paths(found_entities[0].id, found_entities[1].id, max_hops=3)
            if paths:
                results[0]["paths"] = paths
        
        return {"query": query, "results": results}
    except Exception as e:
        logger.error(f"KG query error: {e}")
        return {"error": str(e), "results": []}


@tool
def kg_reasoning(query: str) -> Dict[str, Any]:
    """
    Perform multi-hop reasoning over the medical knowledge graph.
    Example: 'If patient has chest pain and high blood pressure, what specialist should they see?'
    """
    try:
        entities = entity_extractor.extract(query)
        result = reasoner.reason(query, entities)
        return {
            "query": query,
            "conclusion": result.conclusion,
            "steps": [s.model_dump() for s in result.steps],
            "confidence": result.confidence,
        }
    except Exception as e:
        logger.error(f"KG reasoning error: {e}")
        return {"error": str(e)}


# ── Medical KG Tool delegation ────────────────────────────────────────

class ToMedicalKnowledgeGraph:
    """
    Delegate to the Medical Knowledge Graph agent for:
    - Disease-symptom-medication relationship queries
    - Multi-hop medical reasoning (e.g., 'what specialist for these symptoms?')
    - Drug interaction and contraindication analysis
    - Building and querying patient-specific medical knowledge graphs
    """
    def __init__(self, query: str):
        self.query = query


# ── KG Agent Node ─────────────────────────────────────────────────────

kg_agent_tools = [
    extract_medical_entities,
    build_medical_relations,
    query_medical_kg,
    kg_reasoning,
]

KG_AGENT_RUNNABLE = None  # Created lazily


def _get_kg_runnable():
    global KG_AGENT_RUNNABLE
    if KG_AGENT_RUNNABLE is None:
        from customer_support_chat.app.services.assistants.assistant_base import Assistant, CompleteOrEscalate
        KG_AGENT_RUNNABLE = Assistant(
            [ToMedicalKnowledgeGraph],
            kg_agent_tools,
            "medical_kg",
            """You are a Medical Knowledge Graph specialist. Your role:

1. EXTRACT medical entities (diseases, symptoms, medications, doctors, hospitals) from user queries
2. BUILD relationships between extracted entities (HAS_SYMPTOM, TREATS, PRESCRIBED_BY, etc.)
3. QUERY the knowledge graph to find related entities, drug interactions, and multi-hop paths
4. REASON over the graph for complex medical questions (e.g., differential diagnosis paths)

Use the tools provided to:
- extract_medical_entities: Identify medical terms in text
- build_medical_relations: Build connections between entities and store them
- query_medical_kg: Find related entities and paths
- kg_reasoning: Multi-hop reasoning over the graph

Always cite your sources. If you don't have enough information, say so clearly.
If the task is complete or you cannot help, use CompleteOrEscalate to return control.
DO NOT provide definitive medical diagnoses — always recommend consulting a doctor.
""",
        )
    return KG_AGENT_RUNNABLE


async def kg_agent_node(state, config=None):
    """LangGraph node for the Medical Knowledge Graph agent."""
    runnable = _get_kg_runnable()
    result = await runnable.agent.ainvoke(state, config=config)
    return result


def route_kg_agent(state):
    """Route after KG agent: either use tools or go back to primary."""
    from langgraph.prebuilt import tools_condition
    route = tools_condition(state)
    return route


def create_kg_agent():
    """Factory to create or retrieve the KG agent."""
    return _get_kg_runnable()
