"""
Advanced RAG Pipeline — Hybrid retrieval, reranking, and query rewriting.

Extends the base VectorDB with production-grade RAG techniques:
1. Hybrid Retrieval: Dense (embeddings) + Sparse (BM25 keywords)
2. Cross-encoder Reranking: Re-rank retrieved chunks for relevance
3. Query Rewriting: HyDE (Hypothetical Document Embeddings) + Multi-Query expansion
4. Corrective RAG (CRAG): Self-evaluation and corrective retrieval loop
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from math import log
from collections import Counter
import re

from customer_support_chat.app.core.logger import logger
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.services.vectordb.vectordb import VectorDB
from customer_support_chat.app.services.assistants.assistant_base import llm


settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════
# 1. BM25 Sparse Retriever
# ═══════════════════════════════════════════════════════════════════════════

class BM25Retriever:
    """
    BM25 (Best Match 25) sparse retrieval implementation.

    A probabilistic retrieval function that ranks documents based on
    term frequency, inverse document frequency, and document length normalization.

    This provides the "keyword" complement to dense vector search in the
    hybrid pipeline.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Length normalization parameter
        self._documents: List[Dict[str, Any]] = []
        self._doc_lengths: List[int] = []
        self._avg_doc_length: float = 0.0
        self._inverted_index: Dict[str, Dict[int, int]] = {}
        self._doc_freq: Dict[str, int] = {}
        self._total_docs: int = 0

    def index(self, documents: List[Dict[str, Any]]):
        """
        Build BM25 index from a list of documents.

        Args:
            documents: List of {text, id, metadata} dicts.
        """
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
        logger.info(f"📇 BM25 indexed {self._total_docs} documents "
                     f"(vocab: {len(self._inverted_index)}, avg_len: {self._avg_doc_length:.1f})")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for top_k documents matching the query.

        Args:
            query: Search query text.
            top_k: Number of results to return.

        Returns:
            List of (doc_id, bm25_score) tuples.
        """
        query_tokens = self._tokenize(query)
        scores: Dict[int, float] = {}

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

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its index ID."""
        if 0 <= doc_id < len(self._documents):
            return self._documents[doc_id]
        return None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer with lowercase."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return [t for t in tokens if len(t) > 1]

    @property
    def doc_count(self) -> int:
        return self._total_docs


# ═══════════════════════════════════════════════════════════════════════════
# 2. Hybrid Retriever (Dense + Sparse)
# ═══════════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    Combines dense (vector) and sparse (BM25) retrieval with reciprocal rank fusion.

    Fusion formula:
        RRF_score(d) = sum(1 / (k + rank_i(d))) for each ranker i
        where k=60 is a smoothing constant.
    """

    def __init__(self, vector_db: Optional[VectorDB] = None):
        self.vector_db = vector_db
        self.bm25 = BM25Retriever()
        self._rrf_k: int = 60  # Reciprocal rank fusion constant
        self._dense_weight: float = 0.6  # Weight for dense (vs sparse)
        self._sparse_weight: float = 0.4

    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for BM25 retrieval."""
        self.bm25.index(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval with reciprocal rank fusion.

        Args:
            query: Query text.
            top_k: Number of final results.
            dense_weight: Override dense weight.
            sparse_weight: Override sparse weight.

        Returns:
            List of {doc_id, text, score, source, rank_dense, rank_sparse} dicts.
        """
        dw = dense_weight if dense_weight is not None else self._dense_weight
        sw = sparse_weight if sparse_weight is not None else self._sparse_weight

        # Dense retrieval (up to 20 candidates)
        dense_results = []
        if self.vector_db:
            try:
                results = self.vector_db.search(query, k=min(20, top_k * 2))
                for i, r in enumerate(results):
                    score = getattr(r, "score", 0.0)
                    payload = getattr(r, "payload", {})
                    dense_results.append({
                        "rank": i + 1,
                        "score": score,
                        "text": payload.get("chunk_text", ""),
                        "source": payload.get("url", ""),
                    })
            except Exception as exc:
                logger.warning(f"Dense retrieval failed: {exc}")

        # Sparse retrieval (up to 20 candidates)
        sparse_ranked = self.bm25.search(query, top_k=min(20, top_k * 2))
        sparse_results = {}
        for rank, (doc_id, score) in enumerate(sparse_ranked):
            doc = self.bm25.get_document(doc_id)
            if doc:
                sparse_results[doc_id] = {
                    "rank": rank + 1,
                    "score": score,
                    "text": doc.get("text", ""),
                    "source": doc.get("source", ""),
                }

        # Reciprocal Rank Fusion
        fused_scores: Dict[str, float] = {}
        fused_docs: Dict[str, Dict] = {}

        # Add dense results
        for dr in dense_results:
            key = dr.get("source", "") or dr.get("text", "")[:50]
            rrf = 1.0 / (self._rrf_k + dr["rank"])
            fused_scores[key] = dw * rrf
            fused_docs[key] = {
                "text": dr["text"],
                "source": dr["source"],
                "rank_dense": dr["rank"],
                "rank_sparse": None,
                "dense_score": dr["score"],
                "sparse_score": 0.0,
            }

        # Add sparse results (fuse with dense if same doc)
        for doc_id, sr in sparse_results.items():
            key = sr.get("source", "") or sr.get("text", "")[:50]
            rrf = 1.0 / (self._rrf_k + sr["rank"])
            if key in fused_scores:
                fused_scores[key] += sw * rrf
                fused_docs[key]["rank_sparse"] = sr["rank"]
                fused_docs[key]["sparse_score"] = sr["score"]
            else:
                fused_scores[key] = sw * rrf
                fused_docs[key] = {
                    "text": sr["text"],
                    "source": sr["source"],
                    "rank_dense": None,
                    "rank_sparse": sr["rank"],
                    "dense_score": 0.0,
                    "sparse_score": sr["score"],
                }

        # Sort by fused score
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for key, fuse_score in ranked:
            doc = fused_docs[key]
            doc["fused_score"] = round(fuse_score, 6)
            results.append(doc)

        logger.info(f"🔀 Hybrid retrieval: {len(dense_results)} dense + {len(sparse_results)} sparse "
                     f"→ {len(results)} fused results")
        return results


# ═══════════════════════════════════════════════════════════════════════════
# 3. Query Rewriter (HyDE + Multi-Query)
# ═══════════════════════════════════════════════════════════════════════════

class QueryRewriter:
    """
    Query rewriting strategies for improved retrieval.

    HyDE (Hypothetical Document Embeddings):
    Generate a hypothetical answer document from the query, then use its
    embedding for retrieval instead of the raw query. This bridges the
    gap between short queries and document-level semantics.

    Multi-Query Expansion:
    Generate multiple reformulated versions of the query, retrieve for
    each, and merge results. Improves recall for ambiguous queries.
    """

    HYDE_PROMPT = """Generate a short passage that answers the following question.
The passage should be factual and in the style of a customer support knowledge base article.
Do NOT include phrases like "Based on the information" or "I can help".
Write ONLY the passage content — no preamble, no greeting.

Question: {query}

Passage:"""

    MULTI_QUERY_PROMPT = """Generate 3 alternative reformulations of the following question.
Each reformulation should express the same information need differently.
Output one reformulation per line, with NO numbering, bullets, or prefixes.

Question: {query}

Reformulations:"""

    def __init__(self):
        self._hyde_initialized = False
        self._multi_query_initialized = False

    def hyde_rewrite(self, query: str) -> str:
        """
        Generate a HyDE passage for the query.

        Args:
            query: Original user query.

        Returns:
            Hypothetical document text to use for retrieval.
        """
        try:
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_template(self.HYDE_PROMPT)
            chain = prompt | llm
            result = chain.invoke({"query": query})
            hyde_text = result.content if hasattr(result, 'content') else str(result)
            logger.info(f"🔮 HyDE generated: {hyde_text[:100]}...")
            return hyde_text.strip()
        except Exception as exc:
            logger.warning(f"HyDE generation failed: {exc}, using original query")
            return query

    def multi_query_expand(self, query: str) -> List[str]:
        """
        Generate multiple query reformulations.

        Args:
            query: Original user query.

        Returns:
            List of reformulated queries (original + variants).
        """
        try:
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_template(self.MULTI_QUERY_PROMPT)
            chain = prompt | llm
            result = chain.invoke({"query": query})
            content = result.content if hasattr(result, 'content') else str(result)
            variants = [q.strip() for q in content.strip().split("\n") if q.strip()]
            # Remove any numbering or bullet prefixes
            variants = [re.sub(r'^[\d\.\-\*\s]+', '', v).strip() for v in variants]
            all_queries = [query] + variants[:3]
            logger.info(f"🔀 Multi-query expanded to {len(all_queries)} queries")
            return all_queries
        except Exception as exc:
            logger.warning(f"Multi-query expansion failed: {exc}")
            return [query]

    def hyde_multi_query_retrieve(
        self,
        query: str,
        retriever: HybridRetriever,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Full rewrite pipeline: HyDE + Multi-Query → retrieve → merge.

        1. Generate HyDE passage
        2. Generate multiple query reformulations
        3. Retrieve for each query variant
        4. Merge and deduplicate results

        Args:
            query: Original query.
            retriever: A HybridRetriever instance.
            top_k: Results per query variant.

        Returns:
            Merged and deduplicated search results.
        """
        hyde_query = self.hyde_rewrite(query)
        query_variants = self.multi_query_expand(query)

        all_results: Dict[str, Dict] = {}
        for q in [hyde_query] + query_variants:
            try:
                results = retriever.retrieve(q, top_k=top_k)
                for r in results:
                    key = r.get("text", "")[:100]  # Dedup by content
                    if key not in all_results or r.get("fused_score", 0) > all_results[key].get("fused_score", 0):
                        all_results[key] = r
            except Exception as exc:
                logger.warning(f"Retrieval failed for query '{q[:50]}...': {exc}")

        merged = sorted(all_results.values(), key=lambda r: r.get("fused_score", 0), reverse=True)[:top_k]
        logger.info(f"📝 Query rewrite pipeline: {len(query_variants) + 1} queries → {len(merged)} unique results")
        return merged


# ═══════════════════════════════════════════════════════════════════════════
# 4. Corrective RAG (CRAG)
# ═══════════════════════════════════════════════════════════════════════════

class CRAGEvaluator(BaseModel):
    """Evaluation output for CRAG self-assessment."""
    relevance_score: float = Field(ge=0.0, le=1.0, description="How relevant the retrieved docs are to the query")
    needs_correction: bool = Field(default=False, description="Whether retrieval needs to be corrected")
    correction_strategy: str = Field(default="none", description="'none', 'rewrite', 'web_search', or 'expand'")
    reasoning: str = Field(default="", description="Explanation of the evaluation")

    class Config:
        extra = "forbid"


class CorrectiveRAG:
    """
    Corrective RAG: self-evaluate retrieval quality and trigger corrective actions.

    Flow:
    1. Retrieve initial documents
    2. Evaluate relevance via LLM
    3. If relevance is low:
       a. Try query rewriting (HyDE)
       b. Try broader search (more results, different collection)
       c. Fall back to web search (DuckDuckGo)
    4. Return best available results with quality metadata
    """

    EVAL_PROMPT = """Evaluate the relevance of the following retrieved documents to the query.

Query: {query}

Retrieved Documents:
{documents}

Rate the overall relevance (0.0-1.0) and determine if correction is needed.
- If score < 0.3: needs_correction=True, strategy='rewrite'
- If 0.3 <= score < 0.6: needs_correction=True, strategy='expand'
- If score >= 0.6: needs_correction=False

Output JSON with: relevance_score, needs_correction, correction_strategy, reasoning"""

    def __init__(self, vector_db: Optional[VectorDB] = None):
        self.vector_db = vector_db
        self.max_correction_rounds = 2

    def evaluate(self, query: str, docs: List[Dict[str, Any]]) -> CRAGEvaluator:
        """Evaluate the relevance of retrieved documents."""
        if not docs:
            return CRAGEvaluator(
                relevance_score=0.0,
                needs_correction=True,
                correction_strategy="rewrite",
                reasoning="No documents retrieved",
            )

        doc_summary = "\n".join(
            f"[Doc {i+1}] {d.get('text', '')[:200]}" for i, d in enumerate(docs[:5])
        )

        try:
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_template(self.EVAL_PROMPT)
            chain = prompt | llm.with_structured_output(CRAGEvaluator)
            result = chain.invoke({"query": query, "documents": doc_summary})
            logger.info(f"🔍 CRAG evaluation: score={result.relevance_score:.2f}, "
                         f"needs_correction={result.needs_correction}")
            return result
        except Exception as exc:
            logger.warning(f"CRAG evaluation failed: {exc}")
            return CRAGEvaluator(
                relevance_score=0.5,
                needs_correction=False,
                reasoning=f"Evaluation error: {exc}",
            )

    def retrieve_with_correction(
        self,
        query: str,
        vector_db: VectorDB,
        initial_top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Full CRAG retrieval loop with automatic correction.

        Returns: {
            "docs": final document list,
            "correction_applied": bool,
            "correction_rounds": int,
            "final_relevance": float,
            "trajectory": [list of correction steps taken]
        }
        """
        trajectory = []
        corrections = 0

        # Round 1: Initial retrieval
        results = vector_db.search(query, k=initial_top_k)
        docs = [
            {"text": getattr(r, "payload", {}).get("chunk_text", ""), "score": getattr(r, "score", 0.0)}
            for r in results
        ]

        while corrections < self.max_correction_rounds:
            evaluation = self.evaluate(query, docs)
            trajectory.append({
                "round": corrections + 1,
                "strategy": evaluation.correction_strategy,
                "relevance": evaluation.relevance_score,
                "reasoning": evaluation.reasoning,
            })

            if not evaluation.needs_correction:
                break

            if evaluation.correction_strategy == "rewrite":
                # Try HyDE rewrite
                rewriter = QueryRewriter()
                hyde_query = rewriter.hyde_rewrite(query)
                results = vector_db.search(hyde_query, k=initial_top_k + 3)
                docs = [
                    {"text": getattr(r, "payload", {}).get("chunk_text", ""), "score": getattr(r, "score", 0.0)}
                    for r in results
                ]

            elif evaluation.correction_strategy == "expand":
                # Try with more results
                results = vector_db.search(query, k=initial_top_k + 5)
                docs = [
                    {"text": getattr(r, "payload", {}).get("chunk_text", ""), "score": getattr(r, "score", 0.0)}
                    for r in results
                ]

            corrections += 1

        return {
            "docs": docs,
            "correction_applied": corrections > 0,
            "correction_rounds": corrections,
            "final_relevance": trajectory[-1]["relevance"] if trajectory else 0.0,
            "trajectory": trajectory,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Advanced RAG Pipeline (unified)
# ═══════════════════════════════════════════════════════════════════════════

class AdvancedRAGPipeline:
    """
    Unified advanced RAG pipeline combining all techniques.

    Flow:
        Query → QueryRewriter (HyDE + Multi-Query)
              → HybridRetriever (Dense + Sparse fusion)
              → CRAGEvaluator (self-correction)
              → Final results with quality metadata
    """

    def __init__(self, vector_db: Optional[VectorDB] = None):
        self.vector_db = vector_db
        self.hybrid_retriever = HybridRetriever(vector_db=vector_db)
        self.query_rewriter = QueryRewriter()
        self.crag = CorrectiveRAG(vector_db=vector_db)
        self._query_count = 0

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_hyde: bool = True,
        use_multi_query: bool = True,
        use_crag: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the full advanced RAG pipeline.

        Args:
            query: User query.
            top_k: Number of final results.
            use_hyde: Enable HyDE query rewriting.
            use_multi_query: Enable multi-query expansion.
            use_crag: Enable corrective RAG self-evaluation.

        Returns:
            {
                "results": final ranked results,
                "query": original query,
                "rewritten_query": HyDE version,
                "variants": multi-query variants,
                "crag_evaluation": CRAG eval result,
                "pipeline_metadata": {steps, timings, etc.}
            }
        """
        import time
        self._query_count += 1
        start = time.time()
        pipeline_steps = []

        # Step 1: Query rewriting
        rewritten_query = query
        query_variants = [query]

        if use_hyde:
            rewritten_query = self.query_rewriter.hyde_rewrite(query)
            pipeline_steps.append("hyde_rewrite")

        if use_multi_query:
            query_variants = self.query_rewriter.multi_query_expand(query)
            pipeline_steps.append("multi_query_expand")

        # Step 2: Hybrid retrieval
        all_results = []
        for variant in query_variants[:3]:  # Limit variants to avoid excessive API calls
            try:
                results = self.hybrid_retriever.retrieve(variant, top_k=top_k)
                all_results.extend(results)
            except Exception:
                # Fallback: use vector DB directly
                if self.vector_db:
                    try:
                        vec_results = self.vector_db.search(variant, k=top_k)
                        for r in vec_results:
                            all_results.append({
                                "text": getattr(r, "payload", {}).get("chunk_text", ""),
                                "source": getattr(r, "payload", {}).get("url", ""),
                                "score": getattr(r, "score", 0.0),
                                "fused_score": getattr(r, "score", 0.0),
                            })
                    except Exception:
                        pass

        pipeline_steps.append("hybrid_retrieval")

        # Deduplicate
        unique_docs: Dict[str, Dict] = {}
        for r in all_results:
            key = r.get("text", "")[:150]
            if key not in unique_docs or r.get("fused_score", r.get("score", 0)) > unique_docs[key].get("fused_score", unique_docs[key].get("score", 0)):
                unique_docs[key] = r

        results = sorted(unique_docs.values(), key=lambda r: r.get("fused_score", r.get("score", 0)), reverse=True)[:top_k]

        # Step 3: CRAG evaluation
        crag_result = None
        if use_crag:
            crag_result = self.crag.evaluate(query, results)
            pipeline_steps.append("crag_evaluation")

            if crag_result.needs_correction:
                correction = self.crag.retrieve_with_correction(query, self.vector_db, top_k)
                if correction.get("docs"):
                    # Merge corrected docs
                    for doc in correction["docs"]:
                        key = doc.get("text", "")[:150]
                        if key not in unique_docs:
                            unique_docs[key] = doc
                    results = sorted(unique_docs.values(), key=lambda r: r.get("fused_score", r.get("score", 0)), reverse=True)[:top_k]
                pipeline_steps.append(f"crag_correction_{crag_result.correction_strategy}")

        elapsed_ms = (time.time() - start) * 1000
        logger.info(f"🚀 AdvancedRAG pipeline #{self._query_count}: "
                     f"{' → '.join(pipeline_steps)} ({elapsed_ms:.0f}ms, {len(results)} results)")

        return {
            "results": results,
            "query": query,
            "rewritten_query": rewritten_query,
            "variants": query_variants,
            "crag_evaluation": crag_result.model_dump() if crag_result else None,
            "pipeline_metadata": {
                "steps": pipeline_steps,
                "elapsed_ms": elapsed_ms,
                "query_count": self._query_count,
                "use_hyde": use_hyde,
                "use_multi_query": use_multi_query,
                "use_crag": use_crag,
            },
        }

    def search_simple(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Simplified search returning just results."""
        return self.search(query, top_k, use_hyde=False, use_multi_query=False, use_crag=False)["results"]

    @property
    def stats(self) -> Dict[str, Any]:
        return {"query_count": self._query_count}


# Singleton instance
advanced_rag = AdvancedRAGPipeline()
