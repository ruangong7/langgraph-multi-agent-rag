"""
Graph Reasoning — Multi-hop traversal, path discovery, and graph-based inference.

Supports:
- Multi-hop path finding between entities
- Entity expansion (find related entities for context enrichment)
- Contradiction detection in the graph
- Structured reasoning result formatting for LLM consumption
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field
from customer_support_chat.app.core.logger import logger
from .graph_store import GraphStore, GraphNode, graph_store


# ── Reasoning Result Types ──────────────────────────────────────────────

class ReasoningStep(BaseModel):
    """A single reasoning step."""
    step_number: int
    description: str = Field(description="Human-readable description of this reasoning step")
    source_entity: str = ""
    relation_type: str = ""
    target_entity: str = ""
    confidence: float = 1.0


class ReasoningResult(BaseModel):
    """Result of a graph reasoning operation."""
    query: str = Field(description="Original reasoning query")
    method: str = Field(description="Reasoning method used")
    steps: List[ReasoningStep] = Field(default_factory=list)
    conclusion: str = Field(default="", description="Inferred conclusion")
    supporting_entities: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, description="Overall reasoning confidence")
    graph_context: Dict[str, Any] = Field(default_factory=dict)

    def format_for_llm(self) -> str:
        """Format the reasoning result as context for an LLM."""
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
        if self.supporting_entities:
            lines.extend(["", f"**Related Entities:** {', '.join(self.supporting_entities)}"])
        return "\n".join(lines)


# ── Graph Reasoning Engine ───────────────────────────────────────────────

class GraphReasoning:
    """
    Multi-hop reasoning engine for knowledge graph traversal and inference.

    Methods:
    - find_paths: Find paths between two entities
    - expand_context: Expand from an entity to enrich context
    - check_contradiction: Detect conflicting information in the graph
    - reason_about: General-purpose graph reasoning
    """

    def __init__(self, store: Optional[GraphStore] = None):
        self.store = store or graph_store

    def find_paths(
        self,
        source_value: str,
        target_value: str,
        max_hops: int = 3,
        source_type: Optional[str] = None,
        target_type: Optional[str] = None,
    ) -> ReasoningResult:
        """
        Find all paths between two entities using multi-hop traversal.

        Args:
            source_value: Value or partial value of source entity.
            target_value: Value or partial value of target entity.
            max_hops: Maximum number of hops.
            source_type: Optional entity type filter for source.
            target_type: Optional entity type filter for target.

        Returns:
            ReasoningResult with discovered paths.
        """
        logger.info(f"🔍 Finding paths: '{source_value}' → '{target_value}' (max {max_hops} hops)")

        # Resolve entities
        sources = self.store.find_entities(value_contains=source_value)
        targets = self.store.find_entities(value_contains=target_value)

        if source_type:
            sources = [s for s in sources if s.entity_type == source_type]
        if target_type:
            targets = [t for t in targets if t.entity_type == target_type]

        if not sources:
            return ReasoningResult(
                query=f"Path: {source_value} → {target_value}",
                method="path_finding",
                conclusion=f"No source entity found matching '{source_value}'",
                confidence=0.0,
            )
        if not targets:
            return ReasoningResult(
                query=f"Path: {source_value} → {target_value}",
                method="path_finding",
                conclusion=f"No target entity found matching '{target_value}'",
                confidence=0.0,
            )

        source_id = sources[0].id
        target_id = targets[0].id

        if source_id == target_id:
            return ReasoningResult(
                query=f"Path: {source_value} → {target_value}",
                method="path_finding",
                conclusion="Source and target are the same entity.",
                confidence=1.0,
            )

        # Find all simple paths
        try:
            raw_paths = list(nx.all_simple_paths(
                self.store.graph, source_id, target_id, cutoff=max_hops
            ))
        except Exception:
            raw_paths = []

        if not raw_paths:
            return ReasoningResult(
                query=f"Path: {source_value} → {target_value}",
                method="path_finding",
                conclusion=f"No path found within {max_hops} hops.",
                confidence=0.0,
            )

        # Build reasoning steps from the shortest path
        best_path = raw_paths[0]  # sorted by length
        steps = []
        for i in range(len(best_path) - 1):
            u, v = best_path[i], best_path[i + 1]
            edge_data = self.store.get_relations_between(u, v)
            if edge_data:
                rel = edge_data[0]
                u_node = self.store.get_entity(u)
                v_node = self.store.get_entity(v)
                steps.append(ReasoningStep(
                    step_number=i + 1,
                    description=f"{u_node.value if u_node else '?'} "
                                f"-[{rel['type']}]→ {v_node.value if v_node else '?'}",
                    source_entity=u_node.value if u_node else "",
                    relation_type=rel["type"],
                    target_entity=v_node.value if v_node else "",
                    confidence=rel["confidence"],
                ))

        # Collect all unique entities along all paths
        all_entities = set()
        for path in raw_paths:
            for nid in path:
                node = self.store.get_entity(nid)
                if node:
                    all_entities.add(node.value)

        confidence = 1.0 / len(best_path) if len(best_path) > 1 else 1.0

        return ReasoningResult(
            query=f"Path: {source_value} → {target_value}",
            method="path_finding",
            steps=steps,
            conclusion=f"Found path with {len(steps)} hops: "
                       f"{' → '.join(s.source_entity for s in steps)} → {target_value}",
            supporting_entities=list(all_entities),
            confidence=confidence,
            graph_context={
                "num_paths_found": len(raw_paths),
                "shortest_path_length": len(best_path) - 1,
                "all_paths": [[self.store.get_entity(nid).value if self.store.get_entity(nid) else "?" for nid in p] for p in raw_paths[:5]],
            },
        )

    def expand_context(
        self,
        entity_value: str,
        entity_type: Optional[str] = None,
        hops: int = 2,
    ) -> ReasoningResult:
        """
        Expand from an entity to gather related context.

        This is the core operation for Knowledge Graph RAG:
        given a question entity, find all related entities and their relationships.

        Args:
            entity_value: Starting entity value.
            entity_type: Optional entity type filter.
            hops: Number of expansion hops (1 = direct neighbors, 2 = neighbors of neighbors).

        Returns:
            ReasoningResult with expanded context.
        """
        logger.info(f"🌐 Expanding context from: '{entity_value}' ({hops} hops)")

        entities = self.store.find_entities(value_contains=entity_value)
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]

        if not entities:
            return ReasoningResult(
                query=f"Expand: {entity_value}",
                method="context_expansion",
                conclusion=f"No entity found matching '{entity_value}'",
                confidence=0.0,
            )

        center = entities[0]
        neighbor_data = self.store.get_neighbors(center.id, hops=hops)

        steps = []
        all_entities = {center.value}

        for neighbor in neighbor_data["neighbors"]:
            if neighbor["node"]:
                all_entities.add(neighbor["node"].value)
                for rel in neighbor["relations"]:
                    if rel["source"] and rel["target"]:
                        steps.append(ReasoningStep(
                            step_number=len(steps) + 1,
                            description=f"{rel['source'].value} "
                                         f"-[{rel['type']}]→ {rel['target'].value}",
                            source_entity=rel["source"].value,
                            relation_type=rel["type"],
                            target_entity=rel["target"].value,
                            confidence=rel["confidence"],
                        ))

        # Group relations by type for structured context
        relation_groups: Dict[str, List[str]] = {}
        for step in steps:
            relation_groups.setdefault(step.relation_type, []).append(
                f"{step.source_entity} → {step.target_entity}"
            )

        context_summary = []
        for rel_type, pairs in relation_groups.items():
            context_summary.append(f"- {rel_type}: {', '.join(pairs[:5])}")

        return ReasoningResult(
            query=f"Expand: {entity_value}",
            method="context_expansion",
            steps=steps,
            conclusion=f"Found {len(steps)} relationships across {len(all_entities)} entities "
                       f"within {hops} hops of '{entity_value}'.",
            supporting_entities=sorted(all_entities),
            confidence=min(1.0, len(steps) / 10.0),
            graph_context={
                "center_entity": center.value,
                "expansion_hops": hops,
                "relation_summary": context_summary,
                "neighbor_count": len(neighbor_data["neighbors"]),
            },
        )

    def check_contradiction(
        self,
        entity_value: str,
        property_type: str,
    ) -> ReasoningResult:
        """
        Check for contradictory information about an entity.

        e.g., check if a flight has both "DEPARTS_FROM ZRH" and "DEPARTS_FROM GVA".

        Args:
            entity_value: Entity to check.
            property_type: Relation type to check for contradictions.

        Returns:
            ReasoningResult with contradiction analysis.
        """
        logger.info(f"⚠️ Checking contradictions for: '{entity_value}' [{property_type}]")

        entities = self.store.find_entities(value_contains=entity_value)
        if not entities:
            return ReasoningResult(
                query=f"Contradiction check: {entity_value} [{property_type}]",
                method="contradiction_detection",
                conclusion=f"No entity found matching '{entity_value}'",
                confidence=0.0,
            )

        # Get all same-type outgoing relations
        target_node = entities[0]
        all_relations = self.store.search_entities_by_relation(property_type)

        # Find relations where this entity is the source
        conflicting = []
        for rel in all_relations:
            if rel["source"]["value"] == target_node.value:
                conflicting.append(rel)

        if len(conflicting) <= 1:
            return ReasoningResult(
                query=f"Contradiction check: {entity_value} [{property_type}]",
                method="contradiction_detection",
                conclusion=f"No contradictions detected for '{entity_value}' [{property_type}].",
                confidence=1.0,
            )

        # Multiple same-type relations → potential contradiction
        targets = [c["target"]["value"] for c in conflicting]
        steps = [
            ReasoningStep(
                step_number=i + 1,
                description=f"{entity_value} -[{property_type}]→ {t}",
                source_entity=entity_value,
                relation_type=property_type,
                target_entity=t,
                confidence=c["confidence"],
            )
            for i, (c, t) in enumerate(zip(conflicting, targets))
        ]

        return ReasoningResult(
            query=f"Contradiction check: {entity_value} [{property_type}]",
            method="contradiction_detection",
            steps=steps,
            conclusion=f"⚠️ Found {len(conflicting)} potentially contradictory "
                       f"[{property_type}] relations: {targets}",
            supporting_entities=targets,
            confidence=0.3,  # Low confidence = likely contradictory
            graph_context={"conflict_count": len(conflicting), "conflicting_values": targets},
        )

    def reason_about(
        self,
        query_entities: List[str],
        max_hops: int = 3,
    ) -> ReasoningResult:
        """
        General-purpose reasoning: given a set of query entities, find
        how they are connected in the knowledge graph.

        Args:
            query_entities: List of entity values from the user query.
            max_hops: Maximum hops for connectivity analysis.

        Returns:
            ReasoningResult with connection analysis.
        """
        if len(query_entities) < 2:
            return self.expand_context(query_entities[0], hops=max_hops) if query_entities else ReasoningResult(
                query="Empty entity list", method="general_reasoning",
                conclusion="No entities provided for reasoning.", confidence=0.0
            )

        # Find connectivity between query entities
        resolved = []
        for value in query_entities:
            matches = self.store.find_entities(value_contains=value, max_results=1)
            if matches:
                resolved.append(matches[0])

        if len(resolved) < 2:
            return self.expand_context(query_entities[0], hops=max_hops)

        # Check pairwise connectivity
        steps = []
        all_connected = set()
        for i in range(min(len(resolved), 5)):
            for j in range(i + 1, min(len(resolved), 5)):
                source = resolved[i]
                target = resolved[j]
                pair_result = self.find_paths(
                    source.value, target.value, max_hops=max_hops,
                    source_type=source.entity_type, target_type=target.entity_type,
                )
                if pair_result.steps:
                    steps.extend(pair_result.steps)
                    all_connected.update(pair_result.supporting_entities)

        if not steps:
            return ReasoningResult(
                query=f"Reason about: {query_entities}",
                method="general_reasoning",
                conclusion=f"No connections found between entities {query_entities} within {max_hops} hops.",
                confidence=0.0,
            )

        return ReasoningResult(
            query=f"Reason about: {query_entities}",
            method="general_reasoning",
            steps=steps,
            conclusion=f"Found {len(steps)} connection paths between "
                       f"{len(all_connected)} related entities.",
            supporting_entities=list(all_connected),
            confidence=min(1.0, len(steps) / 5.0),
            graph_context={
                "query_entities": query_entities,
                "resolved_count": len(resolved),
                "connected_pairs": len(steps),
            },
        )

    def get_entity_summary(self, entity_value: str) -> Dict[str, Any]:
        """Get a structured summary of everything known about an entity."""
        entities = self.store.find_entities(value_contains=entity_value)
        if not entities:
            return {"found": False, "value": entity_value}

        node = entities[0]
        neighbors = self.store.get_neighbors(node.id, hops=1)

        relation_groups: Dict[str, List[str]] = {}
        for n in neighbors["neighbors"]:
            if n["node"]:
                for rel in n["relations"]:
                    key = rel["type"]
                    relation_groups.setdefault(key, []).append(
                        rel["target"].value if rel["source"] == node else rel["source"].value
                    )

        return {
            "found": True,
            "value": node.value,
            "type": node.entity_type,
            "first_seen": node.first_seen,
            "occurrence_count": node.occurrence_count,
            "relations": relation_groups,
            "total_relations": neighbors.get("subgraph", {}).get("edges", []),
        }


# Import networkx here to avoid circular dependency (used in find_paths)
import networkx as nx

# Singleton instance
graph_reasoning = GraphReasoning()
