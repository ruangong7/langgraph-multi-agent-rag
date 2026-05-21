"""
KG Retriever — Hybrid Knowledge Graph + Vector retrieval for GraphRAG.

Combines:
1. Knowledge Graph context expansion (structured entity relationships)
2. Vector-based semantic search (Qdrant dense retrieval)

This implements a lightweight GraphRAG pattern: use the KG to find related entities,
then use the vector DB to find semantically relevant chunks about those entities.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from customer_support_chat.app.core.logger import logger
from customer_support_chat.app.services.vectordb.vectordb import VectorDB
from .entity_extractor import EntityExtractor, ExtractedEntity, entity_extractor
from .relation_builder import RelationBuilder, relation_builder
from .graph_store import GraphStore, graph_store
from .reasoning import GraphReasoning, ReasoningResult, graph_reasoning


# ── Retrieval Context Types ──────────────────────────────────────────────

class KGContext(BaseModel):
    """Knowledge Graph context for a query."""
    entities: List[ExtractedEntity] = Field(default_factory=list)
    direct_relations: List[Dict[str, Any]] = Field(default_factory=list)
    expanded_context: Optional[ReasoningResult] = None
    entity_summaries: List[Dict[str, Any]] = Field(default_factory=list)


class VectorContext(BaseModel):
    """Vector search results for a query."""
    query: str = ""
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    collection_name: str = ""


class RetrievalContext(BaseModel):
    """Combined retrieval context (KG + Vector) for RAG."""
    kg_context: KGContext = Field(default_factory=KGContext)
    vector_context: VectorContext = Field(default_factory=VectorContext)
    fusion_score: float = Field(default=0.0, description="Combined relevance score")

    def format_for_llm(self) -> str:
        """Format the retrieval context as a prompt for an LLM."""
        parts = ["## Retrieved Context", ""]

        # KG Context
        if self.kg_context.entities:
            parts.append("### 🌐 Knowledge Graph Context")
            parts.append(f"**Entities:** {', '.join(e.value for e in self.kg_context.entities)}")
            if self.kg_context.direct_relations:
                parts.append("**Direct Relations:**")
                for rel in self.kg_context.direct_relations:
                    parts.append(
                        f"  - {rel.get('source', {}).get('value', '?')} "
                        f"-[{rel.get('relation', '?')}]→ "
                        f"{rel.get('target', {}).get('value', '?')}"
                    )
            if self.kg_context.expanded_context and self.kg_context.expanded_context.steps:
                parts.append(f"**Expanded Context:** {self.kg_context.expanded_context.conclusion}")
            parts.append("")

        # Vector Context
        if self.vector_context.chunks:
            parts.append("### 📚 Semantic Search Results")
            for i, chunk in enumerate(self.vector_context.chunks[:5], 1):
                payload = chunk.get("payload", {})
                text = payload.get("chunk_text", "")[:300]
                url = payload.get("url", "")
                score = chunk.get("score", 0)
                parts.append(f"**Chunk {i}** (score: {score:.3f}):")
                parts.append(f"  {text}")
                if url:
                    parts.append(f"  Source: {url}")
                parts.append("")

        return "\n".join(parts) if len(parts) > 2 else "No relevant context found."


# ── KG Retriever ─────────────────────────────────────────────────────────

class KGRetriever:
    """
    Hybrid retriever combining knowledge graph traversal with vector similarity search.

    Flow:
    1. Extract entities from user query
    2. Build relations between entities
    3. Ingest into knowledge graph
    4. Expand context via graph reasoning
    5. Search vector DB using entity-augmented query
    6. Fuse and return combined context
    """

    def __init__(
        self,
        store: Optional[GraphStore] = None,
        vector_db: Optional[VectorDB] = None,
        reasoning: Optional[GraphReasoning] = None,
        extractor: Optional[EntityExtractor] = None,
        builder: Optional[RelationBuilder] = None,
    ):
        self.store = store or graph_store
        self.vector_db = vector_db  # Set lazily to avoid import issues
        self.reasoning = reasoning or graph_reasoning
        self.extractor = extractor or entity_extractor
        self.builder = builder or relation_builder
        self._query_count = 0

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        kg_expansion_hops: int = 2,
        vector_collection: str = "faq_collection",
    ) -> RetrievalContext:
        """
        Hybrid KG + Vector retrieval.

        Args:
            query: User query text.
            top_k: Number of vector chunks to retrieve.
            kg_expansion_hops: Hops for KG context expansion.
            vector_collection: Qdrant collection name for vector search.

        Returns:
            RetrievalContext with combined KG and vector results.
        """
        self._query_count += 1
        logger.info(f"🔎 KGRetriever.retrieve #{self._query_count}: {query[:100]}...")

        # Step 1: Extract entities
        extraction_result = self.extractor.extract(query)
        entities = extraction_result.entities

        # Step 2: Build relations
        relations = self.builder.build_relations(query, entities)

        # Step 3: Ingest into the KG (enrich over time)
        if entities:
            self.store.ingest_batch(entities, relations)

        # Step 4: Expand context via graph reasoning
        direct_relations = [
            {
                "source": {"value": rel.subject.value, "type": rel.subject.entity_type},
                "target": {"value": rel.object.value, "type": rel.object.entity_type},
                "relation": rel.relation_type,
                "confidence": rel.confidence,
            }
            for rel in relations
        ]

        expanded = None
        entity_summaries = []
        if entities:
            # Expand from the most confident entity
            primary = max(entities, key=lambda e: e.confidence) if entities else None
            if primary:
                expanded = self.reasoning.expand_context(
                    primary.value,
                    entity_type=primary.entity_type,
                    hops=kg_expansion_hops,
                )

            # Get summaries for all query entities
            for e in entities[:3]:
                summary = self.reasoning.get_entity_summary(e.value)
                if summary.get("found"):
                    entity_summaries.append(summary)

        kg_context = KGContext(
            entities=entities,
            direct_relations=direct_relations,
            expanded_context=expanded,
            entity_summaries=entity_summaries,
        )

        # Step 5: Vector search with entity-augmented query
        vector_context = self._vector_search(query, entities, top_k, vector_collection)

        # Step 6: Compute fusion score
        fusion_score = self._compute_fusion_score(kg_context, vector_context)

        return RetrievalContext(
            kg_context=kg_context,
            vector_context=vector_context,
            fusion_score=fusion_score,
        )

    def _vector_search(
        self,
        query: str,
        entities: List[ExtractedEntity],
        top_k: int,
        collection: str,
    ) -> VectorContext:
        """
        Perform vector search, optionally augmenting the query with entity information.

        If entities are found, we build an augmented query that includes entity
        values to improve search accuracy (hybrid semantic + entity-aware retrieval).
        """
        if not self.vector_db:
            logger.warning("VectorDB not initialized, skipping vector search")
            return VectorContext(query=query, chunks=[], collection_name=collection)

        # Augment query with entity values for better retrieval
        augmented_query = query
        if entities:
            entity_values = " | ".join(e.value for e in entities[:5])
            augmented_query = f"{query} [Entities: {entity_values}]"

        try:
            results = self.vector_db.search(augmented_query, k=top_k)
            chunks = [
                {
                    "id": getattr(r, "id", str(i)),
                    "score": getattr(r, "score", 0.0),
                    "payload": getattr(r, "payload", {}),
                }
                for i, r in enumerate(results)
            ]
            logger.info(f"📚 Vector search returned {len(chunks)} chunks")
            return VectorContext(query=augmented_query, chunks=chunks, collection_name=collection)
        except Exception as exc:
            logger.error(f"❌ Vector search failed: {exc}")
            return VectorContext(query=augmented_query, chunks=[], collection_name=collection)

    def _compute_fusion_score(self, kg: KGContext, vec: VectorContext) -> float:
        """Compute a combined relevance score from KG and vector contexts."""
        score = 0.0

        # KG contribution: entities found + relation confidence
        if kg.entities:
            score += min(0.3, len(kg.entities) * 0.1)
        if kg.direct_relations:
            avg_conf = sum(r.get("confidence", 0.5) for r in kg.direct_relations) / len(kg.direct_relations)
            score += min(0.3, avg_conf * 0.3)
        if kg.expanded_context and kg.expanded_context.confidence > 0:
            score += kg.expanded_context.confidence * 0.2

        # Vector contribution: number of results + average score
        if vec.chunks:
            num_chunks_factor = min(1.0, len(vec.chunks) / 5.0)
            avg_score = sum(c.get("score", 0) for c in vec.chunks) / len(vec.chunks)
            score += 0.2 * num_chunks_factor + 0.2 * avg_score

        return min(1.0, score)

    def batch_ingest_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        Batch ingest documents into the knowledge graph.

        Args:
            documents: List of {text, source} dicts.

        Returns:
            Total number of relations ingested.
        """
        total_relations = 0
        for doc in documents:
            text = doc.get("text", "")
            if not text:
                continue
            entities = self.extractor.extract(text).entities
            if entities:
                relations = self.builder.build_relations(text, entities)
                self.store.ingest_batch(entities, relations)
                total_relations += len(relations)

        logger.info(f"📥 Batch ingested {len(documents)} docs → {total_relations} relations")
        return total_relations

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "query_count": self._query_count,
            "graph_stats": self.store.get_statistics(),
        }


# Singleton instance (vector_db injected at initialization when Qdrant is available)
kg_retriever = KGRetriever()
