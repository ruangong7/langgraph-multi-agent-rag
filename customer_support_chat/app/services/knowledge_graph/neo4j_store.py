"""
Neo4j Graph Store Adapter — Production-grade knowledge graph persistence with Neo4j.

Provides a drop-in alternative to the NetworkX-based GraphStore, using Neo4j
for scalable graph storage, Cypher queries, and graph algorithms.

Usage:
    from customer_support_chat.app.services.knowledge_graph.neo4j_store import Neo4jGraphStore
    store = Neo4jGraphStore(uri="bolt://localhost:7687", user="neo4j", password="password")
    store.add_entity(extracted_entity)
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
from collections import defaultdict

from customer_support_chat.app.core.logger import logger

# Optional Neo4j import — gracefully handles missing driver
try:
    from neo4j import GraphDatabase, Driver, Session, Result
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("neo4j driver not installed. Neo4jGraphStore will not be available. "
                   "Install with: pip install neo4j")

from .entity_extractor import ExtractedEntity
from .relation_builder import EntityRelation


# ── Entity resolver (shared with NetworkX store) ────────────────────────

def _make_entity_id(entity_type: str, value: str) -> str:
    raw = f"{entity_type}:{value.lower().strip()}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── Cypher Query Templates ──────────────────────────────────────────────

CYPHER_MERGE_NODE = """
MERGE (n:{entity_type} {{id: $id}})
ON CREATE SET
    n.value = $value,
    n.raw_text = $raw_text,
    n.confidence = $confidence,
    n.first_seen = datetime($first_seen),
    n.last_seen = datetime($last_seen),
    n.occurrence_count = 1
ON MATCH SET
    n.last_seen = datetime($last_seen),
    n.occurrence_count = n.occurrence_count + 1,
    n.confidence = CASE
        WHEN $confidence > n.confidence THEN $confidence
        ELSE n.confidence
    END
RETURN n
"""

CYPHER_MERGE_EDGE = """
MATCH (a {{id: $source_id}})
MATCH (b {{id: $target_id}})
MERGE (a)-[r:{relation_type}]->(b)
ON CREATE SET
    r.confidence = $confidence,
    r.evidence = $evidence,
    r.first_seen = datetime($first_seen),
    r.weight = $weight
ON MATCH SET
    r.confidence = CASE
        WHEN $confidence > r.confidence THEN $confidence
        ELSE r.confidence
    END,
    r.weight = r.weight + $weight
RETURN r
"""

CYPHER_GET_ENTITY = """
MATCH (n {id: $id})
RETURN n
"""

CYPHER_FIND_ENTITIES_BY_TYPE = """
MATCH (n:{entity_type})
RETURN n
ORDER BY n.confidence DESC
LIMIT $limit
"""

CYPHER_FIND_ENTITIES_BY_VALUE = """
MATCH (n)
WHERE toLower(n.value) CONTAINS toLower($value)
RETURN n
ORDER BY n.confidence DESC
LIMIT $limit
"""

CYPHER_GET_NEIGHBORS = """
MATCH (n {id: $id})-[r]-(neighbor)
RETURN n, r, neighbor
LIMIT $limit
"""

CYPHER_FIND_PATHS = """
MATCH path = (start {id: $source_id})-[*1..{max_hops}]-(end {id: $target_id})
RETURN path
LIMIT $limit
"""

CYPHER_RELATION_SEARCH = """
MATCH (a)-[r:{relation_type}]->(b)
RETURN a, r, b
LIMIT $limit
"""

CYPHER_GRAPH_STATS = """
MATCH (n)
WITH count(n) AS total_nodes
MATCH ()-[r]->()
RETURN total_nodes, count(r) AS total_edges
"""

CYPHER_ENTITY_TYPE_STATS = """
MATCH (n)
RETURN labels(n)[0] AS entity_type, count(n) AS count
ORDER BY count DESC
"""

CYPHER_RELATION_TYPE_STATS = """
MATCH ()-[r]->()
RETURN type(r) AS relation_type, count(r) AS count
ORDER BY count DESC
"""

CYPHER_CENTRALITY = """
MATCH (n)-[r]-()
WITH n, count(r) AS degree
RETURN n.value AS entity, n.id AS id, degree
ORDER BY degree DESC
LIMIT $limit
"""

CYPHER_CLEAR = """
MATCH (n)
DETACH DELETE n
"""

CYPHER_DELETE_ENTITY = """
MATCH (n {id: $id})
DETACH DELETE n
"""


# ── Neo4j Graph Store ───────────────────────────────────────────────────

class Neo4jGraphStore:
    """
    Neo4j-backed knowledge graph store.

    Features:
    - Transactional entity & relation management
    - Cypher-based graph queries with parameterized inputs
    - Multi-hop path finding via variable-length patterns
    - Automatic node/edge property updates on duplicate
    - Graph statistics via Cypher aggregations
    - Connection pooling via the official Neo4j Python driver

    Requires:
        pip install neo4j
        A running Neo4j instance (local or Neo4j AuraDB cloud)
    """

    DEFAULT_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    DEFAULT_USER = os.environ.get("NEO4J_USER", "neo4j")
    DEFAULT_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "Neo4j driver not installed. Install with: pip install neo4j"
            )

        self.uri = uri or self.DEFAULT_URI
        self.user = user or self.DEFAULT_USER
        self.password = password or self.DEFAULT_PASSWORD
        self._driver: Optional[Driver] = None
        self._connected = False
        self._connect()

    def _connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
            )
            # Verify connection
            self._driver.verify_connectivity()
            self._connected = True
            logger.info(f"🔗 Connected to Neo4j at {self.uri}")
        except (ServiceUnavailable, AuthError) as exc:
            self._connected = False
            logger.error(f"❌ Failed to connect to Neo4j at {self.uri}: {exc}")
            raise ConnectionError(f"Cannot connect to Neo4j: {exc}")

    def _session(self) -> Session:
        """Get a new Neo4j session."""
        if not self._driver:
            raise RuntimeError("Neo4j driver not initialized")
        return self._driver.session()

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver:
            self._driver.close()
            self._connected = False
            logger.info("🔌 Neo4j connection closed")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── CRUD Operations ──────────────────────────────────────────────

    def add_entity(self, entity: ExtractedEntity) -> str:
        """
        Add or update an entity node in Neo4j.

        Uses Cypher MERGE for idempotent upsert:
        - On CREATE: initializes all properties
        - On MATCH: updates last_seen and increments occurrence_count
        """
        node_id = _make_entity_id(entity.entity_type, entity.value)
        now = datetime.now().isoformat()

        with self._session() as session:
            result = session.run(
                CYPHER_MERGE_NODE,
                id=node_id,
                entity_type=entity.entity_type,
                value=entity.value,
                raw_text=entity.raw_text,
                confidence=entity.confidence,
                first_seen=now,
                last_seen=now,
            )
            record = result.single()
            logger.debug(f"📌 Neo4j upsert: [{entity.entity_type}] {entity.value}")

        return node_id

    def add_relation(self, relation: EntityRelation) -> Tuple[str, str]:
        """
        Add or update a relationship edge in Neo4j.

        Uses Cypher MERGE for idempotent edge upsert.
        Both subject and object entities are automatically created if they don't exist.
        """
        source_id = self.add_entity(relation.subject)
        target_id = self.add_entity(relation.object)
        now = datetime.now().isoformat()

        with self._session() as session:
            session.run(
                CYPHER_MERGE_EDGE,
                source_id=source_id,
                target_id=target_id,
                relation_type=relation.relation_type,
                confidence=relation.confidence,
                evidence=relation.evidence,
                first_seen=now,
                weight=relation.confidence,
            )
            logger.debug(f"🔗 Neo4j edge: {relation.subject.value} "
                         f"-[{relation.relation_type}]→ {relation.object.value}")

        return (source_id, target_id)

    def ingest_batch(
        self,
        entities: List[ExtractedEntity],
        relations: List[EntityRelation],
    ) -> Dict[str, int]:
        """Batch ingest entities and relations in a single transaction."""
        with self._session() as session:
            with session.begin_transaction() as tx:
                for entity in entities:
                    node_id = _make_entity_id(entity.entity_type, entity.value)
                    now = datetime.now().isoformat()
                    tx.run(
                        CYPHER_MERGE_NODE,
                        id=node_id,
                        entity_type=entity.entity_type,
                        value=entity.value,
                        raw_text=entity.raw_text,
                        confidence=entity.confidence,
                        first_seen=now,
                        last_seen=now,
                    )

                for relation in relations:
                    source_id = _make_entity_id(relation.subject.entity_type, relation.subject.value)
                    target_id = _make_entity_id(relation.object.entity_type, relation.object.value)
                    now = datetime.now().isoformat()
                    tx.run(
                        CYPHER_MERGE_EDGE,
                        source_id=source_id,
                        target_id=target_id,
                        relation_type=relation.relation_type,
                        confidence=relation.confidence,
                        evidence=relation.evidence,
                        first_seen=now,
                        weight=relation.confidence,
                    )

            stats = self.get_statistics()
            logger.info(f"📥 Neo4j batch ingested: {stats}")
            return {
                "entities_processed": len(entities),
                "relations_processed": len(relations),
                "total_nodes": stats.get("total_nodes", 0),
                "total_edges": stats.get("total_edges", 0),
            }

    def remove_entity(self, node_id: str) -> bool:
        """Remove an entity and all its relationships."""
        with self._session() as session:
            result = session.run(CYPHER_DELETE_ENTITY, id=node_id)
            summary = result.consume()
            deleted = summary.counters.nodes_deleted > 0
            if deleted:
                logger.info(f"🗑️ Neo4j deleted entity: {node_id}")
            return deleted

    # ── Query Operations ─────────────────────────────────────────────

    def get_entity(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get entity properties by node ID."""
        with self._session() as session:
            result = session.run(CYPHER_GET_ENTITY, id=node_id)
            record = result.single()
            if record:
                return dict(record["n"])
        return None

    def find_entities(
        self,
        entity_type: Optional[str] = None,
        value_contains: Optional[str] = None,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """Find entities by type and/or value substring."""
        with self._session() as session:
            if entity_type and not value_contains:
                result = session.run(
                    CYPHER_FIND_ENTITIES_BY_TYPE,
                    entity_type=entity_type,
                    limit=max_results,
                )
                return [dict(record["n"]) for record in result]
            elif value_contains:
                result = session.run(
                    CYPHER_FIND_ENTITIES_BY_VALUE,
                    value=value_contains,
                    limit=max_results,
                )
                entities = [dict(record["n"]) for record in result]
                if entity_type:
                    entities = [e for e in entities
                                if list(e.get("labels", []))[0] == entity_type]
                return entities
            return []

    def get_neighbors(
        self,
        node_id: str,
        hops: int = 1,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get neighbors of an entity within k hops."""
        with self._session() as session:
            result = session.run(CYPHER_GET_NEIGHBORS, id=node_id, limit=limit)
            neighbors = []
            for record in result:
                neighbors.append({
                    "node": dict(record["neighbor"]),
                    "relation": {
                        "type": record["r"].type,
                        "properties": dict(record["r"]),
                    },
                })
            return neighbors

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3,
        limit: int = 10,
    ) -> List[List[Dict]]:
        """Find all paths between two entities up to max_hops."""
        with self._session() as session:
            result = session.run(
                CYPHER_FIND_PATHS,
                source_id=source_id,
                target_id=target_id,
                max_hops=max_hops,
                limit=limit,
            )
            paths = []
            for record in result:
                path = record["path"]
                path_data = []
                for node in path.nodes:
                    path_data.append({
                        "id": node.get("id"),
                        "value": node.get("value"),
                        "type": list(node.labels)[0] if node.labels else "Unknown",
                    })
                # Also include relationships
                for rel in path.relationships:
                    path_data.append({
                        "relation_type": rel.type,
                        "confidence": rel.get("confidence", 1.0),
                    })
                paths.append(path_data)
            return paths

    def search_entities_by_relation(
        self,
        relation_type: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Find all entity pairs connected by a specific relation type."""
        with self._session() as session:
            result = session.run(
                CYPHER_RELATION_SEARCH,
                relation_type=relation_type,
                limit=limit,
            )
            return [
                {
                    "source": dict(record["a"]),
                    "target": dict(record["b"]),
                    "relation": relation_type,
                }
                for record in result
            ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph-level statistics."""
        with self._session() as session:
            # Total counts
            result = session.run(CYPHER_GRAPH_STATS)
            record = result.single()
            total_nodes = record["total_nodes"] if record else 0
            total_edges = record["total_edges"] if record else 0

            # Entity type distribution
            result = session.run(CYPHER_ENTITY_TYPE_STATS)
            entity_types = {record["entity_type"]: record["count"] for record in result}

            # Relation type distribution
            result = session.run(CYPHER_RELATION_TYPE_STATS)
            relation_types = {record["relation_type"]: record["count"] for record in result}

            # Top central entities
            result = session.run(CYPHER_CENTRALITY, limit=5)
            centrality = {
                record["entity"]: record["degree"]
                for record in result
            }

            return {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "entity_types": entity_types,
                "relation_types": relation_types,
                "top_central_entities": centrality,
                "backend": "neo4j",
            }

    def clear(self) -> None:
        """Delete all nodes and relationships."""
        with self._session() as session:
            session.run(CYPHER_CLEAR)
            logger.info("🗑️ Neo4j graph cleared")

    def run_cypher(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Execute an arbitrary Cypher query.

        WARNING: Only use with parameterized queries to prevent injection.

        Args:
            query: Cypher query string with $param placeholders.
            params: Parameter dict.

        Returns:
            List of result records as dicts.
        """
        with self._session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]


# ── Graph Store Factory ─────────────────────────────────────────────────

def create_graph_store(
    backend: str = "networkx",
    **kwargs,
) -> Union[Any, Neo4jGraphStore]:
    """
    Factory function to create the appropriate graph store.

    Args:
        backend: "networkx" or "neo4j"
        **kwargs: Backend-specific arguments (e.g., neo4j_uri, store_path)

    Returns:
        GraphStore or Neo4jGraphStore instance.
    """
    if backend == "neo4j":
        return Neo4jGraphStore(
            uri=kwargs.get("neo4j_uri"),
            user=kwargs.get("neo4j_user"),
            password=kwargs.get("neo4j_password"),
        )
    else:
        from .graph_store import GraphStore
        return GraphStore(store_path=kwargs.get("store_path"))
