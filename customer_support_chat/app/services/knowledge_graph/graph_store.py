"""
Graph Store — NetworkX-based knowledge graph storage, persistence, and querying.

Supports:
- Entity (node) and relation (edge) storage with metadata
- JSON-based persistence for simplicity (no external DB needed)
- Subgraph extraction, neighbor traversal, and centrality analysis
- Visual graph export (Mermaid, Graphviz-compatible DOT format)
"""

import json
import os
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple, Iterator
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict

import networkx as nx
from customer_support_chat.app.core.logger import logger
from .entity_extractor import ExtractedEntity
from .relation_builder import EntityRelation


# ── Core Data Structures ────────────────────────────────────────────────

@dataclass
class GraphNode:
    """A node (entity) in the knowledge graph."""
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
    """An edge (relation) in the knowledge graph."""
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


# ── Graph Store ──────────────────────────────────────────────────────────

class GraphStore:
    """
    NetworkX-based knowledge graph store with JSON persistence.

    Features:
    - Add/update/remove nodes and edges
    - Subgraph extraction by entity type or relation type
    - K-hop neighbor traversal
    - Graph statistics and centrality measures
    - JSON serialization for portability
    - Singleton-like persistent backing file
    """

    DEFAULT_STORE_PATH = "kg_store.json"

    def __init__(self, store_path: Optional[str] = None):
        self._graph = nx.MultiDiGraph()
        self._store_path = store_path or GraphStore.DEFAULT_STORE_PATH
        self._node_registry: Dict[str, GraphNode] = {}
        self._edge_count = 0

        # Load existing store if available
        if os.path.exists(self._store_path):
            self._load()

    # ── CRUD Operations ──────────────────────────────────────────────

    def add_entity(self, entity: ExtractedEntity) -> str:
        """Add an entity as a node. Returns node_id."""
        node = GraphNode.from_entity(entity)
        node_id = node.id

        if node_id in self._node_registry:
            # Update existing node
            existing = self._node_registry[node_id]
            existing.last_seen = datetime.now().isoformat()
            existing.occurrence_count += 1
            existing.confidence = max(existing.confidence, entity.confidence)
            if entity.attributes:
                existing.attributes.update(entity.attributes)
            self._update_graph_node(existing)
        else:
            self._node_registry[node_id] = node
            self._graph.add_node(node_id, **node.to_dict())

        logger.debug(f"📌 Added/updated entity node: [{entity.entity_type}] {entity.value}")
        return node_id

    def add_relation(self, relation: EntityRelation) -> Tuple[str, str]:
        """Add a relation as an edge. Returns (source_id, target_id)."""
        source_id = self.add_entity(relation.subject)
        target_id = self.add_entity(relation.object)
        edge = GraphEdge.from_relation(relation)
        edge.source_id = source_id
        edge.target_id = target_id

        edge_key = f"{source_id}:{relation.relation_type}:{target_id}"
        self._graph.add_edge(source_id, target_id, key=edge_key, **edge.to_dict())
        self._edge_count += 1

        # Also add inverse if bidirectional
        if relation.bidirectional:
            inv_relation = relation.reverse()
            inv_edge = GraphEdge.from_relation(inv_relation)
            inv_edge.source_id = target_id
            inv_edge.target_id = source_id
            inv_key = f"{target_id}:{inv_relation.relation_type}:{source_id}"
            self._graph.add_edge(target_id, source_id, key=inv_key, **inv_edge.to_dict())
            self._edge_count += 1

        logger.debug(f"🔗 Added relation: {relation.subject.value} "
                     f"-[{relation.relation_type}]→ {relation.object.value}")
        return (source_id, target_id)

    def ingest_batch(
        self,
        entities: List[ExtractedEntity],
        relations: List[EntityRelation],
    ) -> Dict[str, int]:
        """
        Ingest a batch of entities and relations.

        Returns dict with counts: {nodes_added, nodes_updated, edges_added}.
        """
        nodes_before = len(self._node_registry)
        edges_before = self._edge_count

        for entity in entities:
            self.add_entity(entity)

        for relation in relations:
            self.add_relation(relation)

        nodes_added = len(self._node_registry) - nodes_before
        edges_added = self._edge_count - edges_before

        stats = {
            "nodes_added": nodes_added,
            "nodes_updated": len(entities) - nodes_added,
            "edges_added": edges_added,
            "total_nodes": self.node_count,
            "total_edges": self.edge_count,
        }
        logger.info(f"📥 Ingested batch: {stats}")
        return stats

    def remove_entity(self, node_id: str) -> bool:
        """Remove an entity and all its incident edges."""
        if node_id in self._node_registry:
            self._graph.remove_node(node_id)
            del self._node_registry[node_id]
            self._edge_count = self._graph.number_of_edges()
            return True
        return False

    # ── Query Operations ─────────────────────────────────────────────

    def get_entity(self, node_id: str) -> Optional[GraphNode]:
        """Get entity by node ID."""
        return self._node_registry.get(node_id)

    def find_entities(
        self,
        entity_type: Optional[str] = None,
        value_contains: Optional[str] = None,
        max_results: int = 50,
    ) -> List[GraphNode]:
        """Find entities by type and/or value substring."""
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

    def get_neighbors(
        self,
        node_id: str,
        hops: int = 1,
        relation_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get k-hop neighbors of a node.

        Returns {
            "entity": GraphNode,
            "neighbors": [{node, relations, path_length}],
            "subgraph": {nodes, edges}
        }
        """
        entity = self.get_entity(node_id)
        if not entity:
            return {"entity": None, "neighbors": [], "subgraph": {"nodes": [], "edges": []}}

        neighbors_result = []
        for target_id, paths in nx.single_source_shortest_path(
            self._graph, node_id, cutoff=hops
        ).items():
            if target_id == node_id:
                continue

            # Collect relations along the path
            path_relations = []
            for i in range(len(paths) - 1):
                s, t = paths[i], paths[i + 1]
                edge_data = self._graph.get_edge_data(s, t)
                if edge_data:
                    for key, data in edge_data.items():
                        rel_type = data.get("relation_type", "RELATED_TO")
                        if not relation_types or rel_type in relation_types:
                            path_relations.append({
                                "source": self._node_registry.get(s),
                                "target": self._node_registry.get(t),
                                "type": rel_type,
                                "confidence": data.get("confidence", 1.0),
                            })

            if path_relations:
                neighbors_result.append({
                    "node": self._node_registry.get(target_id),
                    "relations": path_relations,
                    "path_length": len(paths) - 1,
                })

        return {
            "entity": entity,
            "neighbors": sorted(neighbors_result, key=lambda n: n["path_length"]),
            "subgraph": self._extract_subgraph(node_id, hops),
        }

    def get_relations_between(
        self,
        source_id: str,
        target_id: str,
    ) -> List[Dict[str, Any]]:
        """Get all relations between two entities."""
        edge_data = self._graph.get_edge_data(source_id, target_id)
        if not edge_data:
            return []
        return [
            {"type": d.get("relation_type"), "confidence": d.get("confidence"), "evidence": d.get("evidence")}
            for _, d in edge_data.items()
        ]

    def search_entities_by_relation(
        self,
        relation_type: str,
        entity_value: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find entities connected by a specific relation type.

        Example: search_entities_by_relation("LOCATED_IN") → all (hotel, city) pairs.
        """
        results = []
        for u, v, data in self._graph.edges(data=True):
            if data.get("relation_type") == relation_type:
                u_node = self._node_registry.get(u)
                v_node = self._node_registry.get(v)
                if not u_node or not v_node:
                    continue
                if entity_value:
                    if entity_value.lower() not in u_node.value.lower() and \
                       entity_value.lower() not in v_node.value.lower():
                        continue
                results.append({
                    "source": u_node.to_dict(),
                    "target": v_node.to_dict(),
                    "relation": relation_type,
                    "confidence": data.get("confidence", 1.0),
                })
        return results

    def _extract_subgraph(self, center_id: str, hops: int) -> Dict[str, List]:
        """Extract ego-subgraph centered at a node."""
        subgraph_nodes = list(nx.single_source_shortest_path_length(
            self._graph, center_id, cutoff=hops
        ).keys())
        subgraph_edges = []
        for u in subgraph_nodes:
            for v in subgraph_nodes:
                if self._graph.has_edge(u, v):
                    for _, data in self._graph[u][v].items():
                        subgraph_edges.append({
                            "source": self._node_registry.get(u),
                            "target": self._node_registry.get(v),
                            "type": data.get("relation_type"),
                            "confidence": data.get("confidence"),
                        })
        return {
            "nodes": [self._node_registry[nid].to_dict() for nid in subgraph_nodes if nid in self._node_registry],
            "edges": subgraph_edges,
        }

    def _update_graph_node(self, node: GraphNode) -> None:
        """Update node attributes in the NetworkX graph."""
        if node.id in self._graph:
            for attr, val in node.to_dict().items():
                self._graph.nodes[node.id][attr] = val

    # ── Graph Statistics ─────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph-level statistics."""
        entity_type_counts = defaultdict(int)
        for node in self._node_registry.values():
            entity_type_counts[node.entity_type] += 1

        relation_type_counts = defaultdict(int)
        for _, _, data in self._graph.edges(data=True):
            relation_type_counts[data.get("relation_type", "RELATED_TO")] += 1

        components = list(nx.weakly_connected_components(self._graph))

        # Degree centrality for top nodes
        centrality = {}
        if self._graph.number_of_nodes() > 0:
            deg_centrality = nx.degree_centrality(self._graph)
            top_5 = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            for nid, score in top_5:
                node = self._node_registry.get(nid)
                if node:
                    centrality[node.value] = round(score, 4)

        return {
            "total_nodes": self.node_count,
            "total_edges": self.edge_count,
            "entity_types": dict(entity_type_counts),
            "relation_types": dict(relation_type_counts),
            "connected_components": len(components),
            "largest_component_size": max(len(c) for c in components) if components else 0,
            "density": round(nx.density(self._graph), 6),
            "top_central_entities": centrality,
        }

    # ── Persistence ──────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the graph to JSON."""
        data = {
            "nodes": [n.to_dict() for n in self._node_registry.values()],
            "edges": [
                {**data, "source": u, "target": v}
                for u, v, data in self._graph.edges(data=True)
            ],
            "saved_at": datetime.now().isoformat(),
        }
        with open(self._store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Graph persisted to {self._store_path} "
                     f"({self.node_count} nodes, {self.edge_count} edges)")

    def _load(self) -> None:
        """Load the graph from JSON."""
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
                edge_key = f"{source}:{edge_data.get('relation_type', '')}:{target}"
                self._graph.add_edge(source, target, key=edge_key, **edge_data)
                self._edge_count += 1

            logger.info(f"📂 Loaded graph: {self.node_count} nodes, {self.edge_count} edges "
                         f"(saved {data.get('saved_at', 'unknown')})")
        except Exception as exc:
            logger.warning(f"⚠️ Could not load graph store: {exc}, starting fresh")

    def clear(self) -> None:
        """Clear all data from the store."""
        self._graph.clear()
        self._node_registry.clear()
        self._edge_count = 0

    # ── Export ───────────────────────────────────────────────────────

    def export_mermaid(self) -> str:
        """Export graph as Mermaid flowchart syntax for visualization."""
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

    def export_cytoscape(self) -> Dict[str, List]:
        """Export graph in Cytoscape.js-compatible format."""
        elements = []
        for nid, node in self._node_registry.items():
            elements.append({
                "data": {
                    "id": nid,
                    "label": node.value,
                    "type": node.entity_type,
                },
            })
        for u, v, data in self._graph.edges(data=True):
            elements.append({
                "data": {
                    "id": f"{u}_{v}_{data.get('relation_type')}",
                    "source": u,
                    "target": v,
                    "label": data.get("relation_type", ""),
                    "confidence": data.get("confidence", 1.0),
                },
            })
        return elements

    # ── Properties ───────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return len(self._node_registry)

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    @property
    def entities(self) -> List[GraphNode]:
        return list(self._node_registry.values())

    @property
    def graph(self) -> nx.MultiDiGraph:
        return self._graph


# Singleton store instance
graph_store = GraphStore()
