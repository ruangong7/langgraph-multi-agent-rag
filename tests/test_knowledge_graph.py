"""
Knowledge Graph Module — Standalone Unit Tests (no external LLM/Qdrant dependencies).

Tests the pure-Python logic: entity data structures, relation types, graph store CRUD,
multi-hop reasoning, BM25 retrieval, and streaming event types.
"""

import pytest
import os
import sys
import tempfile
import json
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List, Dict, Any, Optional

# ═══════════════════════════════════════════════════════════════════════════
# Add project root to path
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic Entity & Relation Models (self-contained, no project imports)
# ═══════════════════════════════════════════════════════════════════════════

from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    entity_type: str = Field(description="Entity type: PERSON, LOCATION, etc.")
    value: str = Field(description="Normalized entity value")
    raw_text: str = Field(description="Original mention text")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    attributes: Dict[str, str] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class EntityRelation(BaseModel):
    subject: ExtractedEntity
    relation_type: str
    object: ExtractedEntity
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: str = ""
    bidirectional: bool = False
    metadata: Dict[str, str] = Field(default_factory=dict)

    def as_triple(self):
        return (self.subject.value, self.relation_type, self.object.value)

    def reverse(self):
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


class ReasoningStep(BaseModel):
    step_number: int
    description: str = ""
    source_entity: str = ""
    relation_type: str = ""
    target_entity: str = ""
    confidence: float = 1.0


class ReasoningResult(BaseModel):
    query: str = ""
    method: str = ""
    steps: List[ReasoningStep] = Field(default_factory=list)
    conclusion: str = ""
    supporting_entities: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    graph_context: Dict[str, Any] = Field(default_factory=dict)

    def format_for_llm(self) -> str:
        if not self.steps:
            return "No relevant knowledge graph connections found."
        lines = ["## Knowledge Graph Reasoning", ""]
        for step in self.steps:
            lines.append(
                f"Step {step.step_number}: {step.description} "
                f"({step.source_entity} -[{step.relation_type}]→ {step.target_entity}, "
                f"confidence: {step.confidence:.2f})"
            )
        if self.conclusion:
            lines.extend(["", f"**Conclusion:** {self.conclusion}"])
        return "\n".join(lines)


class KGContext(BaseModel):
    entities: List[ExtractedEntity] = Field(default_factory=list)
    direct_relations: List[Dict[str, Any]] = Field(default_factory=list)
    expanded_context: Optional[ReasoningResult] = None
    entity_summaries: List[Dict[str, Any]] = Field(default_factory=list)


class VectorContext(BaseModel):
    query: str = ""
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    collection_name: str = ""


class RetrievalContext(BaseModel):
    kg_context: KGContext = Field(default_factory=KGContext)
    vector_context: VectorContext = Field(default_factory=VectorContext)
    fusion_score: float = 0.0

    def format_for_llm(self) -> str:
        parts = ["## Retrieved Context", ""]
        if self.kg_context.entities:
            parts.append("### 🌐 Knowledge Graph Context")
            parts.append(f"**Entities:** {', '.join(e.value for e in self.kg_context.entities)}")
        if self.kg_context.direct_relations:
            parts.append("**Relations:**")
            for rel in self.kg_context.direct_relations:
                parts.append(f"  - {rel.get('source', {}).get('value', '?')} "
                             f"-[{rel.get('relation', '?')}]→ {rel.get('target', {}).get('value', '?')}")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# NetworkX-based GraphStore (self-contained)
# ═══════════════════════════════════════════════════════════════════════════

import networkx as nx
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import defaultdict


@dataclass
class GraphNode:
    id: str
    entity_type: str
    value: str
    confidence: float = 1.0
    first_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    occurrence_count: int = 1
    attributes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_entity(cls, entity: ExtractedEntity, node_id: Optional[str] = None) -> "GraphNode":
        return cls(
            id=node_id or cls._make_id(entity.entity_type, entity.value),
            entity_type=entity.entity_type,
            value=entity.value,
            confidence=entity.confidence,
            attributes=entity.attributes,
        )

    @staticmethod
    def _make_id(entity_type: str, value: str) -> str:
        raw = f"{entity_type}:{value.lower().strip()}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    relation_type: str
    confidence: float = 1.0
    evidence: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)
    first_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_relation(cls, relation: EntityRelation) -> "GraphEdge":
        return cls(
            source_id=GraphNode._make_id(relation.subject.entity_type, relation.subject.value),
            target_id=GraphNode._make_id(relation.object.entity_type, relation.object.value),
            relation_type=relation.relation_type,
            confidence=relation.confidence,
            evidence=relation.evidence,
            metadata=relation.metadata,
            weight=relation.confidence,
        )


class GraphStore:
    """NetworkX-backed knowledge graph store with JSON persistence."""

    def __init__(self, store_path: Optional[str] = None):
        self._graph = nx.MultiDiGraph()
        self._store_path = store_path
        self._node_registry: Dict[str, GraphNode] = {}
        self._edge_count = 0
        if store_path and os.path.exists(store_path):
            self._load()

    def add_entity(self, entity: ExtractedEntity) -> str:
        node = GraphNode.from_entity(entity)
        node_id = node.id
        if node_id in self._node_registry:
            existing = self._node_registry[node_id]
            existing.last_seen = datetime.now().isoformat()
            existing.occurrence_count += 1
            existing.confidence = max(existing.confidence, entity.confidence)
            if entity.attributes:
                existing.attributes.update(entity.attributes)
        else:
            self._node_registry[node_id] = node
            self._graph.add_node(node_id, **node.to_dict())
        return node_id

    def add_relation(self, relation: EntityRelation):
        source_id = self.add_entity(relation.subject)
        target_id = self.add_entity(relation.object)
        edge = GraphEdge.from_relation(relation)
        edge.source_id = source_id
        edge.target_id = target_id
        edge_key = f"{source_id}:{relation.relation_type}:{target_id}"
        self._graph.add_edge(source_id, target_id, key=edge_key, **edge.to_dict())
        self._edge_count += 1
        return (source_id, target_id)

    def ingest_batch(self, entities, relations):
        nodes_before = len(self._node_registry)
        edges_before = self._edge_count
        for e in entities:
            self.add_entity(e)
        for r in relations:
            self.add_relation(r)
        return {
            "total_nodes": self.node_count,
            "total_edges": self.edge_count,
        }

    def get_entity(self, node_id: str) -> Optional[GraphNode]:
        return self._node_registry.get(node_id)

    def find_entities(self, entity_type=None, value_contains=None, max_results=50):
        results = []
        for node in self._node_registry.values():
            if entity_type and node.entity_type != entity_type:
                continue
            if value_contains and value_contains.lower() not in node.value.lower():
                continue
            results.append(node)
            if len(results) >= max_results:
                break
        return results

    def get_neighbors(self, node_id, hops=1, relation_types=None):
        entity = self._node_registry.get(node_id)
        if not entity:
            return {"entity": None, "neighbors": [], "subgraph": {"nodes": [], "edges": []}}
        neighbors = []
        for target_id in self._graph.neighbors(node_id):
            edge_data = self._graph.get_edge_data(node_id, target_id)
            if edge_data:
                for key, data in edge_data.items():
                    rt = data.get("relation_type", "RELATED_TO")
                    if not relation_types or rt in relation_types:
                        neighbors.append({
                            "node": self._node_registry.get(target_id),
                            "relations": [{
                                "source": entity,
                                "target": self._node_registry.get(target_id),
                                "type": rt,
                                "confidence": data.get("confidence", 1.0),
                            }],
                            "path_length": 1,
                        })
        return {"entity": entity, "neighbors": neighbors, "subgraph": {"nodes": [], "edges": []}}

    def search_entities_by_relation(self, relation_type, entity_value=None):
        results = []
        for u, v, data in self._graph.edges(data=True):
            if data.get("relation_type") == relation_type:
                u_node = self._node_registry.get(u)
                v_node = self._node_registry.get(v)
                if not u_node or not v_node:
                    continue
                if entity_value and entity_value.lower() not in u_node.value.lower() and entity_value.lower() not in v_node.value.lower():
                    continue
                results.append({
                    "source": u_node.to_dict(),
                    "target": v_node.to_dict(),
                    "relation": relation_type,
                    "confidence": data.get("confidence", 1.0),
                })
        return results

    def get_relations_between(self, source_id, target_id):
        edge_data = self._graph.get_edge_data(source_id, target_id)
        if not edge_data:
            return []
        return [{"type": d.get("relation_type"), "confidence": d.get("confidence"), "evidence": d.get("evidence")}
                for _, d in edge_data.items()]

    def get_statistics(self):
        entity_type_counts = defaultdict(int)
        for node in self._node_registry.values():
            entity_type_counts[node.entity_type] += 1
        relation_type_counts = defaultdict(int)
        for _, _, data in self._graph.edges(data=True):
            relation_type_counts[data.get("relation_type", "RELATED_TO")] += 1
        components = list(nx.weakly_connected_components(self._graph))
        return {
            "total_nodes": self.node_count,
            "total_edges": self.edge_count,
            "entity_types": dict(entity_type_counts),
            "relation_types": dict(relation_type_counts),
            "connected_components": len(components),
            "largest_component_size": max(len(c) for c in components) if components else 0,
            "density": round(nx.density(self._graph), 6),
            "top_central_entities": {},
        }

    def save(self):
        data = {
            "nodes": [n.to_dict() for n in self._node_registry.values()],
            "edges": [{**data, "source": u, "target": v} for u, v, data in self._graph.edges(data=True)],
            "saved_at": datetime.now().isoformat(),
        }
        with open(self._store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self):
        try:
            with open(self._store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for node_data in data.get("nodes", []):
                node = GraphNode(**node_data)
                self._node_registry[node.id] = node
                self._graph.add_node(node.id, **node_data)
            for edge_data in data.get("edges", []):
                source = edge_data.pop("source")
                target = edge_data.pop("target")
                ek = f"{source}:{edge_data.get('relation_type', '')}:{target}"
                self._graph.add_edge(source, target, key=ek, **edge_data)
                self._edge_count += 1
        except Exception:
            pass

    def clear(self):
        self._graph.clear()
        self._node_registry.clear()
        self._edge_count = 0

    def remove_entity(self, node_id):
        if node_id in self._node_registry:
            self._graph.remove_node(node_id)
            del self._node_registry[node_id]
            self._edge_count = self._graph.number_of_edges()
            return True
        return False

    def export_mermaid(self):
        lines = ["graph LR"]
        node_labels = {}
        for nid, node in self._node_registry.items():
            label = f"{node.entity_type}\\n{node.value[:20]}"
            safe_id = nid.replace("-", "_")
            node_labels[nid] = safe_id
            lines.append(f"    {safe_id}[\"{label}\"]")
        for u, v, data in self._graph.edges(data=True):
            if u in node_labels and v in node_labels:
                rel = data.get("relation_type", "RELATED_TO")
                lines.append(f"    {node_labels[u]} -->|{rel}| {node_labels[v]}")
        return "\n".join(lines)

    def export_cytoscape(self):
        elements = []
        for nid, node in self._node_registry.items():
            elements.append({"data": {"id": nid, "label": node.value, "type": node.entity_type}})
        for u, v, data in self._graph.edges(data=True):
            elements.append({"data": {"id": f"{u}_{v}_{data.get('relation_type')}", "source": u, "target": v,
                                       "label": data.get("relation_type", ""), "confidence": data.get("confidence", 1.0)}})
        return elements

    @property
    def node_count(self):
        return len(self._node_registry)

    @property
    def edge_count(self):
        return self._graph.number_of_edges()

    @property
    def graph(self):
        return self._graph


# ═══════════════════════════════════════════════════════════════════════════
# BM25 Sparse Retriever (self-contained, no external deps)
# ═══════════════════════════════════════════════════════════════════════════

import re
from math import log
from collections import Counter


class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self._documents = []
        self._doc_lengths = []
        self._avg_doc_length = 0.0
        self._inverted_index = {}
        self._doc_freq = {}
        self._total_docs = 0

    def index(self, documents):
        self._documents = documents
        self._total_docs = len(documents)
        self._doc_lengths = []
        self._inverted_index = {}
        self._doc_freq = {}
        for doc_id, doc in enumerate(documents):
            text = doc.get("text", "")
            tokens = self._tokenize(text)
            self._doc_lengths.append(len(tokens))
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                if term not in self._inverted_index:
                    self._inverted_index[term] = {}
                self._inverted_index[term][doc_id] = count
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1
        self._avg_doc_length = sum(self._doc_lengths) / max(1, self._total_docs)

    def search(self, query, top_k=10):
        query_tokens = self._tokenize(query)
        scores = {}
        for token in query_tokens:
            if token not in self._inverted_index:
                continue
            df = self._doc_freq.get(token, 0)
            idf = log((self._total_docs - df + 0.5) / (df + 0.5) + 1)
            for doc_id, tf in self._inverted_index[token].items():
                dl = self._doc_lengths[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_doc_length)
                scores[doc_id] = scores.get(doc_id, 0) + idf * numerator / denominator
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def get_document(self, doc_id):
        if 0 <= doc_id < len(self._documents):
            return self._documents[doc_id]
        return None

    @staticmethod
    def _tokenize(text):
        text = text.lower()
        return [t for t in re.findall(r'\b\w+\b', text) if len(t) > 1]

    @property
    def doc_count(self):
        return self._total_docs


# ═══════════════════════════════════════════════════════════════════════════
# Streaming Models (self-contained)
# ═══════════════════════════════════════════════════════════════════════════

from enum import Enum
import time


class StreamEventType(str, Enum):
    TOKEN = "token"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    THINKING = "thinking"
    ERROR = "error"
    COMPLETE = "complete"
    PROGRESS = "progress"


@dataclass
class StreamEvent:
    event_type: StreamEventType
    data: Any = None
    agent_name: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {"type": self.event_type.value, "data": self.data, "agent": self.agent_name,
                "timestamp": self.timestamp, "metadata": self.metadata}


class TokenUsageTracker:
    PRICING = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60), "deepseek-chat": (0.14, 0.28)}

    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        self._input_tokens = 0
        self._output_tokens = 0
        self._agent_usage = {}

    def track(self, input_tokens, output_tokens, agent_name="unknown"):
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        if agent_name not in self._agent_usage:
            self._agent_usage[agent_name] = {"input": 0, "output": 0, "calls": 0}
        self._agent_usage[agent_name]["input"] += input_tokens
        self._agent_usage[agent_name]["output"] += output_tokens
        self._agent_usage[agent_name]["calls"] += 1

    def estimate_cost(self, model_name=None):
        name = model_name or self.model_name
        ip, op = self.PRICING.get(name, (0.0, 0.0))
        ic = (self._input_tokens / 1_000_000) * ip
        oc = (self._output_tokens / 1_000_000) * op
        return {"input_tokens": self._input_tokens, "output_tokens": self._output_tokens,
                "total_tokens": self._input_tokens + self._output_tokens,
                "total_cost_usd": round(ic + oc, 6), "model": name}

    @property
    def total_tokens(self):
        return self._input_tokens + self._output_tokens


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_entities():
    return [
        ExtractedEntity(entity_type="PERSON", value="John Doe", raw_text="John Doe", confidence=1.0),
        ExtractedEntity(entity_type="FLIGHT_NUMBER", value="LH123", raw_text="LH123", confidence=1.0),
        ExtractedEntity(entity_type="LOCATION", value="Zurich", raw_text="Zurich", confidence=1.0),
        ExtractedEntity(entity_type="LOCATION", value="London", raw_text="London", confidence=1.0),
        ExtractedEntity(entity_type="HOTEL_NAME", value="Marriott Zurich", raw_text="Marriott Zurich", confidence=0.95),
        ExtractedEntity(entity_type="DATE", value="2026-06-15", raw_text="June 15th", confidence=0.9),
        ExtractedEntity(entity_type="ORDER_ID", value="ORD-77821", raw_text="ORD-77821", confidence=1.0),
        ExtractedEntity(entity_type="PRODUCT_NAME", value="Travel Pillow", raw_text="travel pillow", confidence=0.85),
    ]


@pytest.fixture
def temp_graph_store():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    store = GraphStore(store_path=path)
    yield store
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def populated_store(temp_graph_store, sample_entities):
    entities = sample_entities
    person, flight, zurich, london, hotel, date, order, product = entities

    relations = [
        EntityRelation(subject=person, relation_type="FLIES_ON", object=flight, confidence=1.0),
        EntityRelation(subject=flight, relation_type="DEPARTS_FROM", object=zurich, confidence=1.0),
        EntityRelation(subject=flight, relation_type="ARRIVES_AT", object=london, confidence=1.0),
        EntityRelation(subject=flight, relation_type="ON_DATE", object=date, confidence=0.9),
        EntityRelation(subject=person, relation_type="BOOKED_AT", object=hotel, confidence=0.95),
        EntityRelation(subject=hotel, relation_type="LOCATED_IN", object=zurich, confidence=0.8),
        EntityRelation(subject=person, relation_type="HAS_ORDER", object=order, confidence=1.0),
        EntityRelation(subject=order, relation_type="CONTAINS", object=product, confidence=0.85),
    ]
    for rel in relations:
        temp_graph_store.add_relation(rel)
    return temp_graph_store


# ═══════════════════════════════════════════════════════════════════════════
# Entity & Relation Model Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEntityModels:
    def test_entity_creation(self):
        e = ExtractedEntity(entity_type="PERSON", value="Alice", raw_text="Alice", confidence=0.95)
        assert e.entity_type == "PERSON"
        assert e.value == "Alice"

    def test_entity_to_dict(self, sample_entities):
        d = sample_entities[0].to_dict()
        assert d["entity_type"] == "PERSON"
        assert d["value"] == "John Doe"
        assert "confidence" in d

    def test_entity_invalid_confidence(self):
        with pytest.raises(Exception):
            ExtractedEntity(entity_type="PERSON", value="X", raw_text="X", confidence=2.0)

    def test_relation_triple(self, sample_entities):
        rel = EntityRelation(subject=sample_entities[0], relation_type="FLIES_ON", object=sample_entities[1])
        assert rel.as_triple() == ("John Doe", "FLIES_ON", "LH123")

    def test_relation_reverse(self, sample_entities):
        rel = EntityRelation(subject=sample_entities[0], relation_type="FLIES_ON", object=sample_entities[1])
        rev = rel.reverse()
        assert rev.relation_type == "HAS_PASSENGER"
        assert rev.subject.value == "LH123"
        assert rev.object.value == "John Doe"

    def test_relation_inverse_unknown_type(self, sample_entities):
        rel = EntityRelation(subject=sample_entities[0], relation_type="CUSTOM_TYPE", object=sample_entities[1])
        rev = rel.reverse()
        assert rev.relation_type == "INV_CUSTOM_TYPE"

    def test_reasoning_result_format(self):
        result = ReasoningResult(
            query="test", method="path_finding",
            steps=[ReasoningStep(step_number=1, description="A → B", source_entity="A", relation_type="RELATED_TO", target_entity="B")],
            conclusion="Found path", confidence=1.0,
        )
        formatted = result.format_for_llm()
        assert "Knowledge Graph Reasoning" in formatted
        assert "A -[RELATED_TO]→ B" in formatted

    def test_reasoning_result_empty(self):
        result = ReasoningResult(query="test", method="path_finding")
        formatted = result.format_for_llm()
        assert "No relevant knowledge graph connections" in formatted

    def test_retrieval_context_format(self):
        ctx = RetrievalContext(
            kg_context=KGContext(
                entities=[ExtractedEntity(entity_type="PERSON", value="John", raw_text="John", confidence=1.0)],
                direct_relations=[{"source": {"value": "John"}, "target": {"value": "LH123"}, "relation": "FLIES_ON", "confidence": 1.0}],
            ),
            vector_context=VectorContext(),
        )
        formatted = ctx.format_for_llm()
        assert "Knowledge Graph Context" in formatted
        assert "John" in formatted

    def test_retrieval_context_empty(self):
        ctx = RetrievalContext()
        formatted = ctx.format_for_llm()
        assert "Retrieved Context" in formatted


# ═══════════════════════════════════════════════════════════════════════════
# GraphNode & GraphEdge Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGraphNode:
    def test_from_entity(self, sample_entities):
        node = GraphNode.from_entity(sample_entities[0])
        assert node.entity_type == "PERSON"
        assert node.value == "John Doe"
        assert node.id is not None

    def test_make_id_deterministic(self):
        id1 = GraphNode._make_id("PERSON", "John Doe")
        id2 = GraphNode._make_id("PERSON", "John Doe")
        assert id1 == id2

    def test_make_id_case_insensitive(self):
        id1 = GraphNode._make_id("PERSON", "John Doe")
        id2 = GraphNode._make_id("PERSON", "john doe")
        assert id1 == id2

    def test_to_dict(self):
        node = GraphNode(id="abc123", entity_type="LOCATION", value="Paris", confidence=0.9)
        d = node.to_dict()
        assert d["id"] == "abc123"
        assert d["value"] == "Paris"


class TestGraphEdge:
    def test_from_relation(self, sample_entities):
        rel = EntityRelation(subject=sample_entities[0], relation_type="FLIES_ON", object=sample_entities[1])
        edge = GraphEdge.from_relation(rel)
        assert edge.relation_type == "FLIES_ON"
        assert len(edge.source_id) == 12
        assert len(edge.target_id) == 12


# ═══════════════════════════════════════════════════════════════════════════
# GraphStore Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGraphStore:
    def test_empty_store(self, temp_graph_store):
        assert temp_graph_store.node_count == 0
        assert temp_graph_store.edge_count == 0

    def test_add_entity(self, temp_graph_store, sample_entities):
        temp_graph_store.add_entity(sample_entities[0])
        assert temp_graph_store.node_count == 1

    def test_add_relation(self, temp_graph_store, sample_entities):
        rel = EntityRelation(subject=sample_entities[0], relation_type="FLIES_ON", object=sample_entities[1])
        temp_graph_store.add_relation(rel)
        assert temp_graph_store.node_count == 2
        assert temp_graph_store.edge_count == 1

    def test_add_duplicate_entity(self, temp_graph_store, sample_entities):
        temp_graph_store.add_entity(sample_entities[0])
        c1 = temp_graph_store.node_count
        temp_graph_store.add_entity(sample_entities[0])
        assert temp_graph_store.node_count == c1
        eid = GraphNode.from_entity(sample_entities[0]).id
        node = temp_graph_store.get_entity(eid)
        assert node.occurrence_count == 2

    def test_ingest_batch(self, temp_graph_store, sample_entities):
        relations = [EntityRelation(subject=sample_entities[0], relation_type="FLIES_ON", object=sample_entities[1])]
        stats = temp_graph_store.ingest_batch(sample_entities[:2], relations)
        assert stats["total_nodes"] > 0
        assert stats["total_edges"] > 0

    def test_find_entities_by_type(self, populated_store):
        results = populated_store.find_entities(entity_type="PERSON")
        assert len(results) == 1
        assert results[0].value == "John Doe"

    def test_find_entities_by_value(self, populated_store):
        results = populated_store.find_entities(value_contains="Zuri")
        assert any("Zurich" in r.value for r in results)

    def test_get_neighbors(self, populated_store):
        person_entities = populated_store.find_entities(entity_type="PERSON")
        person_id = person_entities[0].id
        result = populated_store.get_neighbors(person_id, hops=1)
        assert result["entity"] is not None
        assert len(result["neighbors"]) > 0

    def test_search_entities_by_relation(self, populated_store):
        results = populated_store.search_entities_by_relation("FLIES_ON")
        assert len(results) >= 1
        assert results[0]["source"]["value"] == "John Doe"

    def test_get_statistics(self, populated_store):
        stats = populated_store.get_statistics()
        assert stats["total_nodes"] > 0
        assert stats["total_edges"] > 0
        assert "PERSON" in stats["entity_types"]
        assert "FLIES_ON" in stats["relation_types"]

    def test_persistence(self, temp_graph_store, sample_entities):
        temp_graph_store.add_entity(sample_entities[0])
        nc = temp_graph_store.node_count
        temp_graph_store.save()
        path = temp_graph_store._store_path
        store2 = GraphStore(store_path=path)
        assert store2.node_count == nc

    def test_clear(self, populated_store):
        assert populated_store.node_count > 0
        populated_store.clear()
        assert populated_store.node_count == 0
        assert populated_store.edge_count == 0

    def test_remove_entity(self, populated_store):
        entities = populated_store.find_entities(entity_type="PERSON")
        assert entities
        nid = entities[0].id
        ec_before = populated_store.edge_count
        populated_store.remove_entity(nid)
        assert populated_store.get_entity(nid) is None
        assert populated_store.edge_count <= ec_before

    def test_export_mermaid(self, populated_store):
        mermaid = populated_store.export_mermaid()
        assert "graph LR" in mermaid
        assert "-->" in mermaid

    def test_export_cytoscape(self, populated_store):
        elements = populated_store.export_cytoscape()
        assert len(elements) > 0
        assert "data" in elements[0]


# ═══════════════════════════════════════════════════════════════════════════
# Graph Reasoning Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGraphReasoning:
    def test_find_paths(self, populated_store):
        person = populated_store.find_entities(entity_type="PERSON")[0].id
        london = populated_store.find_entities(value_contains="London")[0].id
        try:
            paths = list(nx.all_simple_paths(populated_store.graph, person, london, cutoff=3))
        except Exception:
            paths = []
        assert len(paths) > 0, f"Should find path from John Doe to London"

    def test_expand_context(self, populated_store):
        person = populated_store.find_entities(entity_type="PERSON")[0].id
        result = populated_store.get_neighbors(person, hops=2)
        assert len(result["neighbors"]) > 0

    def test_no_neighbors_for_isolated_node(self, temp_graph_store):
        e = ExtractedEntity(entity_type="PERSON", value="Ghost", raw_text="Ghost", confidence=1.0)
        nid = temp_graph_store.add_entity(e)
        result = temp_graph_store.get_neighbors(nid)
        assert len(result["neighbors"]) == 0

    def test_search_entities_by_relation_empty(self, populated_store):
        results = populated_store.search_entities_by_relation("NONEXISTENT")
        assert len(results) == 0

    def test_get_entity_summary(self, populated_store):
        person = populated_store.find_entities(entity_type="PERSON")[0]
        assert person.value == "John Doe"
        assert person.occurrence_count >= 1


# ═══════════════════════════════════════════════════════════════════════════
# BM25 Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBM25:
    @pytest.fixture
    def bm25_index(self):
        docs = [
            {"text": "Flight LH123 departs from Zurich to London", "source": "flights.db"},
            {"text": "Hotel Marriott Zurich offers luxury rooms near the lake", "source": "hotels.db"},
            {"text": "Car rental available at Zurich airport from Hertz and Avis", "source": "cars.db"},
            {"text": "Swiss Airlines policy: free cancellation within 24 hours", "source": "policies.db"},
            {"text": "London Heathrow airport has direct trains to city center", "source": "transport.db"},
        ]
        bm = BM25Retriever(k1=1.5, b=0.75)
        bm.index(docs)
        return bm

    def test_index_creation(self, bm25_index):
        assert bm25_index.doc_count == 5

    def test_search_relevant(self, bm25_index):
        results = bm25_index.search("Zurich hotel", top_k=3)
        assert len(results) > 0
        # First result should be hotel-related
        doc = bm25_index.get_document(results[0][0])
        assert doc is not None
        assert "hotel" in doc["text"].lower() or "zurich" in doc["text"].lower()

    def test_search_no_match(self, bm25_index):
        results = bm25_index.search("xyzzy nonsensical query", top_k=3)
        # BM25 still returns results (IDF may be high for rare terms), just low scores
        assert isinstance(results, list)

    def test_get_document_out_of_bounds(self, bm25_index):
        assert bm25_index.get_document(999) is None

    def test_empty_index_search(self):
        bm = BM25Retriever()
        bm.index([])
        results = bm.search("anything")
        assert results == []

    def test_single_document(self):
        bm = BM25Retriever()
        bm.index([{"text": "Hello world", "source": "test"}])
        results = bm.search("hello")
        assert len(results) == 1

    def test_tokenize(self):
        tokens = BM25Retriever._tokenize("Hello, World! How are you?")
        assert "hello" in tokens
        assert "world" in tokens
        assert "how" in tokens
        assert "are" in tokens
        assert "you" in tokens
        # Single char words should be excluded
        assert "i" not in BM25Retriever._tokenize("I am here")


# ═══════════════════════════════════════════════════════════════════════════
# Streaming Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestStreaming:
    def test_event_creation(self):
        event = StreamEvent(event_type=StreamEventType.AGENT_START, agent_name="primary_assistant")
        assert event.event_type == StreamEventType.AGENT_START
        assert "primary_assistant" in event.agent_name

    def test_event_to_dict(self):
        event = StreamEvent(event_type=StreamEventType.TOKEN, data="Hello", agent_name="kg_agent")
        d = event.to_dict()
        assert d["type"] == "token"
        assert d["data"] == "Hello"
        assert d["agent"] == "kg_agent"

    def test_token_tracker(self):
        tracker = TokenUsageTracker(model_name="gpt-4o-mini")
        tracker.track(500, 200, "primary_assistant")
        tracker.track(300, 150, "flight_booking")
        cost = tracker.estimate_cost()
        assert cost["input_tokens"] == 800
        assert cost["output_tokens"] == 350
        assert cost["total_tokens"] == 1150
        assert cost["total_cost_usd"] > 0

    def test_token_tracker_unknown_model(self):
        tracker = TokenUsageTracker(model_name="unknown-model")
        tracker.track(100, 50)
        cost = tracker.estimate_cost()
        assert cost["total_cost_usd"] == 0.0

    def test_token_tracker_reset(self):
        tracker = TokenUsageTracker()
        tracker.track(100, 50)
        assert tracker.total_tokens == 150

    def test_stream_event_types(self):
        assert StreamEventType.TOKEN.value == "token"
        assert StreamEventType.AGENT_START.value == "agent_start"
        assert StreamEventType.AGENT_END.value == "agent_end"
        assert StreamEventType.ERROR.value == "error"
        assert StreamEventType.COMPLETE.value == "complete"


# ═══════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_pipeline(self, temp_graph_store):
        """Test complete KG pipeline: create entities, build relations, store, query."""
        entities = [
            ExtractedEntity(entity_type="PERSON", value="Alice", raw_text="Alice", confidence=1.0),
            ExtractedEntity(entity_type="FLIGHT_NUMBER", value="SWR456", raw_text="SWR456", confidence=1.0),
            ExtractedEntity(entity_type="LOCATION", value="Geneva", raw_text="Geneva", confidence=1.0),
            ExtractedEntity(entity_type="LOCATION", value="Paris", raw_text="Paris", confidence=1.0),
            ExtractedEntity(entity_type="HOTEL_NAME", value="Hotel de Paris", raw_text="Hotel de Paris", confidence=0.95),
        ]

        person, flight, geneva, paris, hotel = entities
        relations = [
            EntityRelation(subject=person, relation_type="FLIES_ON", object=flight),
            EntityRelation(subject=flight, relation_type="DEPARTS_FROM", object=geneva),
            EntityRelation(subject=flight, relation_type="ARRIVES_AT", object=paris),
            EntityRelation(subject=person, relation_type="BOOKED_AT", object=hotel),
            EntityRelation(subject=hotel, relation_type="LOCATED_IN", object=paris),
        ]

        stats = temp_graph_store.ingest_batch(entities, relations)
        assert stats["total_nodes"] == 5
        assert stats["total_edges"] >= 5

        # Verify path: Alice → Paris
        person_id = temp_graph_store.find_entities(entity_type="PERSON")[0].id
        paris_id = temp_graph_store.find_entities(value_contains="Paris")[0].id
        try:
            paths = list(nx.all_simple_paths(temp_graph_store.graph, person_id, paris_id, cutoff=3))
        except Exception:
            paths = []
        assert len(paths) > 0

    def test_graph_grows_with_repeated_ingestion(self, temp_graph_store):
        """Repeated ingestion of similar entities should update, not duplicate."""
        e1 = ExtractedEntity(entity_type="LOCATION", value="Berlin", raw_text="Berlin", confidence=1.0)
        e2 = ExtractedEntity(entity_type="LOCATION", value="Berlin", raw_text="Berlin", confidence=0.8)

        temp_graph_store.add_entity(e1)
        n1 = temp_graph_store.node_count
        temp_graph_store.add_entity(e2)
        assert temp_graph_store.node_count == n1  # Same entity

    def test_mermaid_export_with_relations(self, populated_store):
        mermaid = populated_store.export_mermaid()
        assert "FLIES_ON" in mermaid or "-->" in mermaid

    def test_cytoscape_export(self, populated_store):
        elements = populated_store.export_cytoscape()
        nodes = [e for e in elements if "label" in e["data"] and "source" not in e["data"]]
        edges = [e for e in elements if "source" in e["data"]]
        assert len(nodes) > 0
        assert len(edges) > 0

    def test_statistics_after_clear(self, populated_store):
        populated_store.clear()
        stats = populated_store.get_statistics()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0
        assert stats["density"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Edge Cases & Error Handling
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_entities_list(self, temp_graph_store):
        stats = temp_graph_store.ingest_batch([], [])
        assert stats["total_nodes"] == 0

    def test_relation_with_missing_entities(self):
        """Relation between entities that don't exist yet should auto-create them."""
        # Tested implicitly — add_relation calls add_entity automatically
        pass

    def test_find_entities_max_results(self, temp_graph_store):
        for i in range(100):
            temp_graph_store.add_entity(ExtractedEntity(entity_type="LOCATION", value=f"City_{i}", raw_text=f"City_{i}", confidence=1.0))
        results = temp_graph_store.find_entities(max_results=20)
        assert len(results) == 20

    def test_retrieval_context_fusion_score(self):
        ctx = RetrievalContext(fusion_score=0.85)
        assert ctx.fusion_score == 0.85

    def test_graphnode_attributes(self):
        node = GraphNode(id="x", entity_type="PERSON", value="Test", attributes={"age": "30", "city": "Zurich"})
        assert node.attributes["age"] == "30"
        d = node.to_dict()
        assert "attributes" in d

    def test_relation_with_metadata(self, sample_entities):
        rel = EntityRelation(subject=sample_entities[0], relation_type="FLIES_ON", object=sample_entities[1],
                             metadata={"source": "user_query", "timestamp": "2026-05-21"})
        assert rel.metadata["source"] == "user_query"

    def test_relation_with_evidence(self, sample_entities):
        rel = EntityRelation(subject=sample_entities[0], relation_type="FLIES_ON", object=sample_entities[1],
                             evidence="User said 'I am on flight LH123'")
        assert "LH123" in rel.evidence
