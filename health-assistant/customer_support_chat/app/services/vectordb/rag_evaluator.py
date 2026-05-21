"""
RAG Evaluation Module — Quality assessment for retrieval-augmented generation.

Provides metric computation without requiring external evaluation services:
- Faithfulness: Does the answer stay true to retrieved context?
- Answer Relevance: Does the answer address the query?
- Context Precision: Are retrieved chunks actually relevant?
- Context Recall: Did we retrieve all relevant information?
- Semantic Similarity: How close is the answer to the ground truth?

These metrics follow the RAGAS framework conventions but are implemented
with lightweight, self-contained logic using LLM-as-judge or embedding similarity.
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import re
import math

from customer_support_chat.app.core.logger import logger


# ── Metric Result Types ────────────────────────────────────────────────

class MetricResult(BaseModel):
    """A single evaluation metric result."""
    name: str = Field(description="Metric name (e.g., 'faithfulness', 'context_precision')")
    score: float = Field(ge=0.0, le=1.0, description="Normalized score [0, 1]")
    details: str = Field(default="", description="Explanation of the score")
    passed: bool = Field(default=True, description="Whether the score meets the threshold")


class EvaluationReport(BaseModel):
    """Complete RAG evaluation report."""
    query: str = ""
    answer: str = ""
    contexts: List[str] = Field(default_factory=list)
    ground_truth: str = ""
    metrics: List[MetricResult] = Field(default_factory=list)
    overall_score: float = 0.0
    summary: str = ""

    def to_markdown(self) -> str:
        """Format the report as a Markdown table."""
        lines = [
            "## RAG Evaluation Report",
            "",
            f"**Query:** {self.query[:200]}",
            f"**Answer:** {self.answer[:200]}",
            f"**Contexts:** {len(self.contexts)} chunks",
            "",
            "| Metric | Score | Passed | Details |",
            "|--------|-------|--------|---------|",
        ]
        for m in self.metrics:
            status = "✅" if m.passed else "❌"
            lines.append(f"| {m.name} | {m.score:.3f} | {status} | {m.details[:80]} |")
        lines.extend([
            "",
            f"**Overall Score:** {self.overall_score:.3f}",
            f"**Summary:** {self.summary}",
        ])
        return "\n".join(lines)


# ── Text Utility Functions ─────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase word tokens."""
    return [t.lower() for t in re.findall(r'\b\w+\b', text) if len(t) > 1]


def _ngrams(tokens: List[str], n: int) -> List[str]:
    """Generate n-grams from token list."""
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors."""
    dot = sum(vec1.get(k, 0.0) * vec2.get(k, 0.0) for k in set(vec1) | set(vec2))
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def _term_frequency_vector(text: str) -> Dict[str, float]:
    """Build a TF vector from text."""
    tokens = _tokenize(text)
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0.0) + 1.0
    # Normalize
    total = len(tokens) or 1
    return {k: v / total for k, v in tf.items()}


def _overlap_coefficient(text1: str, text2: str) -> float:
    """Compute overlap coefficient (intersection / min(|A|, |B|))."""
    tokens1 = set(_tokenize(text1))
    tokens2 = set(_tokenize(text2))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    return len(intersection) / min(len(tokens1), len(tokens2))


# ── RAG Metrics ────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Lightweight RAG evaluation metrics for retrieval-augmented generation quality.

    All metrics are self-contained — no external API calls needed.
    Uses lexical overlap and TF-IDF-like vector similarity for scoring.

    Typical usage:
        evaluator = RAGEvaluator(threshold=0.6)
        report = evaluator.evaluate(query, answer, contexts, ground_truth)
        print(report.to_markdown())
    """

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Minimum score for a metric to be considered "passed".
        """
        self.threshold = threshold

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = "",
    ) -> EvaluationReport:
        """
        Run all evaluation metrics and produce a report.

        Args:
            query: The user's original question.
            answer: The generated answer from the RAG system.
            contexts: The retrieved context chunks used for generation.
            ground_truth: Optional reference answer for semantic similarity.

        Returns:
            EvaluationReport with all metrics.
        """
        metrics = []

        # 1. Context Precision: how many retrieved chunks are relevant to the query?
        cp = self.context_precision(query, contexts)
        metrics.append(MetricResult(
            name="context_precision",
            score=cp,
            details=f"{len(contexts)} chunks, precision={cp:.2f}",
            passed=cp >= self.threshold,
        ))

        # 2. Context Recall: how much of the answer's key info appears in contexts?
        cr = self.context_recall(answer, contexts)
        metrics.append(MetricResult(
            name="context_recall",
            score=cr,
            details=f"Recall of answer tokens in contexts: {cr:.2f}",
            passed=cr >= self.threshold,
        ))

        # 3. Faithfulness: is the answer grounded in the contexts?
        faith = self.faithfulness(answer, contexts)
        metrics.append(MetricResult(
            name="faithfulness",
            score=faith,
            details=f"Answer-context overlap: {faith:.2f}",
            passed=faith >= self.threshold,
        ))

        # 4. Answer Relevance: does the answer address the query?
        rel = self.answer_relevance(query, answer)
        metrics.append(MetricResult(
            name="answer_relevance",
            score=rel,
            details=f"Query-answer similarity: {rel:.2f}",
            passed=rel >= self.threshold,
        ))

        # 5. Semantic Similarity (if ground truth provided)
        if ground_truth:
            sim = self.semantic_similarity(answer, ground_truth)
            metrics.append(MetricResult(
                name="semantic_similarity",
                score=sim,
                details=f"Answer vs ground truth: {sim:.2f}",
                passed=sim >= self.threshold,
            ))

        # Compute overall score (weighted average)
        weights = [0.25, 0.25, 0.20, 0.15, 0.15][:len(metrics)]
        # Normalize weights
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        overall = sum(m.score * w for m, w in zip(metrics, weights))

        # Generate summary
        passed_count = sum(1 for m in metrics if m.passed)
        if overall >= 0.8:
            summary = f"✅ Excellent: {passed_count}/{len(metrics)} metrics passed"
        elif overall >= 0.6:
            summary = f"⚠️ Good: {passed_count}/{len(metrics)} metrics passed, room for improvement"
        elif overall >= 0.4:
            summary = f"🔶 Fair: {passed_count}/{len(metrics)} metrics passed, needs work"
        else:
            summary = f"❌ Poor: {passed_count}/{len(metrics)} metrics passed, significant issues"

        return EvaluationReport(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            metrics=metrics,
            overall_score=round(overall, 4),
            summary=summary,
        )

    def context_precision(self, query: str, contexts: List[str]) -> float:
        """
        Context Precision: proportion of retrieved chunks that are relevant to the query.

        Implementation: lexical overlap between query tokens and each chunk,
        averaged over all chunks. Higher = more relevant contexts retrieved.
        """
        if not contexts:
            return 0.0

        precisions = []
        for ctx in contexts:
            overlap = _overlap_coefficient(query, ctx)
            precisions.append(overlap)

        return sum(precisions) / len(precisions) if precisions else 0.0

    def context_recall(self, answer: str, contexts: List[str]) -> float:
        """
        Context Recall: proportion of answer's key information that can be
        found across the retrieved contexts.

        Implementation: max overlap between answer tokens and each context chunk.
        """
        if not contexts or not answer:
            return 0.0

        answer_tokens = set(_tokenize(answer))
        if not answer_tokens:
            return 0.0

        all_context_tokens = set()
        for ctx in contexts:
            all_context_tokens.update(_tokenize(ctx))

        recall = len(answer_tokens & all_context_tokens) / len(answer_tokens)
        return min(1.0, recall)

    def faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Faithfulness: is the answer grounded in the retrieved contexts?

        Implementation: proportion of answer n-grams (1-3) that appear
        in at least one context chunk. Penalizes hallucinated content.
        """
        if not contexts or not answer:
            return 0.0

        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 1.0  # Empty answer is vacuously faithful

        combined_context = " ".join(contexts).lower()

        faithful_count = 0
        total_ngrams = 0

        for n in [1, 2, 3]:
            answer_ngrams = _ngrams(answer_tokens, n)
            for ng in answer_ngrams:
                total_ngrams += 1
                if ng.lower() in combined_context:
                    faithful_count += 1

        if total_ngrams == 0:
            return 0.0
        return faithful_count / total_ngrams

    def answer_relevance(self, query: str, answer: str) -> float:
        """
        Answer Relevance: does the generated answer address the query?

        Implementation: cosine similarity between query TF vector and answer TF vector.
        """
        if not query or not answer:
            return 0.0

        q_vec = _term_frequency_vector(query)
        a_vec = _term_frequency_vector(answer)

        return _cosine_similarity(q_vec, a_vec)

    def semantic_similarity(self, answer: str, ground_truth: str) -> float:
        """
        Semantic Similarity: lexical similarity between answer and ground truth.

        Implementation: TF-IDF-like cosine similarity with n-gram overlap bonus.
        """
        if not answer or not ground_truth:
            return 0.0

        # TF cosine similarity
        a_vec = _term_frequency_vector(answer)
        g_vec = _term_frequency_vector(ground_truth)
        tf_sim = _cosine_similarity(a_vec, g_vec)

        # Bigram overlap bonus
        a_tokens = _tokenize(answer)
        g_tokens = _tokenize(ground_truth)
        a_bigrams = set(_ngrams(a_tokens, 2))
        g_bigrams = set(_ngrams(g_tokens, 2))
        if a_bigrams or g_bigrams:
            bigram_overlap = len(a_bigrams & g_bigrams) / max(len(a_bigrams), len(g_bigrams))
        else:
            bigram_overlap = 0.0

        # Weighted combination
        return 0.7 * tf_sim + 0.3 * bigram_overlap

    def evaluate_retrieval_only(
        self,
        query: str,
        contexts: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate only the retrieval quality (no generation metrics).

        Returns: {context_precision, avg_chunk_length, num_chunks}
        """
        return {
            "context_precision": self.context_precision(query, contexts),
            "avg_chunk_length": sum(len(_tokenize(c)) for c in contexts) / max(1, len(contexts)),
            "num_chunks": len(contexts),
        }


# ── Batch Evaluation ──────────────────────────────────────────────────

class BatchEvaluator:
    """
    Run RAG evaluation across multiple query/answer/context triples
    and produce aggregate statistics.
    """

    def __init__(self, threshold: float = 0.5):
        self.evaluator = RAGEvaluator(threshold=threshold)
        self._reports: List[EvaluationReport] = []

    def add(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = "",
    ) -> EvaluationReport:
        """Evaluate a single example and store the report."""
        report = self.evaluator.evaluate(query, answer, contexts, ground_truth)
        self._reports.append(report)
        return report

    def aggregate(self) -> Dict[str, Any]:
        """
        Compute aggregate metrics across all evaluated examples.

        Returns:
            {metric_name: {mean, std, min, max, count}}
        """
        if not self._reports:
            return {"error": "No evaluations run yet"}

        metric_names = {m.name for r in self._reports for m in r.metrics}

        agg = {}
        for mname in metric_names:
            scores = []
            for r in self._reports:
                for m in r.metrics:
                    if m.name == mname:
                        scores.append(m.score)
            if scores:
                mean = sum(scores) / len(scores)
                variance = sum((s - mean) ** 2 for s in scores) / len(scores)
                agg[mname] = {
                    "mean": round(mean, 4),
                    "std": round(math.sqrt(variance), 4),
                    "min": round(min(scores), 4),
                    "max": round(max(scores), 4),
                    "count": len(scores),
                }

        overall_scores = [r.overall_score for r in self._reports]
        mean_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

        agg["overall"] = {
            "mean": round(mean_overall, 4),
            "std": round(math.sqrt(sum((s - mean_overall) ** 2 for s in overall_scores) / len(overall_scores)), 4),
            "min": round(min(overall_scores), 4),
            "max": round(max(overall_scores), 4),
            "count": len(overall_scores),
        }

        return agg

    def to_summary_markdown(self) -> str:
        """Generate a Markdown summary of aggregate results."""
        agg = self.aggregate()
        lines = [
            "## Batch RAG Evaluation Summary",
            f"**Examples evaluated:** {agg.get('overall', {}).get('count', 0)}",
            "",
            "| Metric | Mean | Std | Min | Max |",
            "|--------|------|-----|-----|-----|",
        ]
        for name, stats in agg.items():
            if name != "overall":
                lines.append(
                    f"| {name} | {stats['mean']:.3f} | {stats['std']:.3f} | "
                    f"{stats['min']:.3f} | {stats['max']:.3f} |"
                )
        if "overall" in agg and isinstance(agg["overall"], dict):
            lines.append(
                f"| **Overall** | **{agg['overall']['mean']:.3f}** | "
                f"{agg['overall']['std']:.3f} | {agg['overall']['min']:.3f} | "
                f"{agg['overall']['max']:.3f} |"
            )
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all stored reports."""
        self._reports.clear()


# ── Singleton ──────────────────────────────────────────────────────────

rag_evaluator = RAGEvaluator()
