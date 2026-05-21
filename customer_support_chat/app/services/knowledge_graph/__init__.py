"""
Knowledge Graph Module for Multi-Agent Customer Support System.

Provides entity extraction, relation building, graph storage,
multi-hop reasoning, GraphRAG hybrid retrieval, and a LangGraph-compatible KG agent.

Architecture:
    entity_extractor  ──▶ relation_builder ──▶ graph_store
                                │                    │
                                ▼                    ▼
                           kg_retriever ◀─── reasoning
                                │
                                ▼
                            kg_agent (LangGraph node)
"""

from .entity_extractor import EntityExtractor, ExtractedEntity
from .relation_builder import RelationBuilder, EntityRelation
from .graph_store import GraphStore, GraphNode, GraphEdge
from .reasoning import GraphReasoning, ReasoningResult
from .kg_retriever import KGRetriever, RetrievalContext
from .kg_agent import create_kg_agent, KGAgentState

__all__ = [
    "EntityExtractor",
    "ExtractedEntity",
    "RelationBuilder",
    "EntityRelation",
    "GraphStore",
    "GraphNode",
    "GraphEdge",
    "GraphReasoning",
    "ReasoningResult",
    "KGRetriever",
    "RetrievalContext",
    "create_kg_agent",
    "KGAgentState",
]
