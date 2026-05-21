"""
Streaming Module — Async streaming, token-level output, and event-based progress.

Adds production-grade streaming to the LangGraph multi-agent system:
1. Token-level streaming via LangChain StreamingHandler
2. Agent reasoning step visibility (show which agent is active)
3. Streaming progress callbacks for web UIs
4. Token usage tracking and cost estimation
"""

import asyncio
import time
from typing import AsyncIterator, Callable, Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import LLMResult

from customer_support_chat.app.core.logger import logger


# ═══════════════════════════════════════════════════════════════════════════
# 1. Streaming Event Types
# ═══════════════════════════════════════════════════════════════════════════

class StreamEventType(str, Enum):
    """Types of streaming events emitted during graph execution."""
    TOKEN = "token"               # Individual LLM token
    AGENT_START = "agent_start"   # An agent node begins processing
    AGENT_END = "agent_end"       # An agent node finishes
    TOOL_START = "tool_start"     # A tool is about to execute
    TOOL_END = "tool_end"         # A tool execution completes
    THINKING = "thinking"         # Agent is reasoning (non-visible thoughts)
    ERROR = "error"               # An error occurred
    COMPLETE = "complete"         # Full response is ready
    PROGRESS = "progress"         # Progress update (e.g., "2/5 agents done")


@dataclass
class StreamEvent:
    """A single streaming event."""
    event_type: StreamEventType
    data: Any = None
    agent_name: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.event_type.value,
            "data": self.data,
            "agent": self.agent_name,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Token-Level Streaming Callback
# ═══════════════════════════════════════════════════════════════════════════

class TokenStreamingHandler(BaseCallbackHandler):
    """
    LangChain callback handler for token-level streaming.

    Captures individual LLM tokens as they are generated and emits them
    via a callback function, enabling real-time token-by-token display.
    """

    def __init__(self, on_token: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        super().__init__()
        self.on_token = on_token
        self._tokens: List[str] = []
        self._token_count: int = 0
        self._start_time: float = 0.0
        self._current_agent: str = ""
        self._token_history: List[Dict[str, Any]] = []

    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs) -> None:
        self._start_time = time.time()
        self._tokens = []
        logger.debug(f"🚀 LLM stream started ({len(prompts)} prompts)")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self._tokens.append(token)
        self._token_count += 1
        if self.on_token:
            try:
                self.on_token(token, {
                    "count": self._token_count,
                    "agent": self._current_agent,
                })
            except Exception:
                pass  # Don't let callback errors break streaming

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        elapsed = time.time() - self._start_time
        tokens_per_second = self._token_count / max(0.001, elapsed)
        full_text = "".join(self._tokens)
        self._token_history.append({
            "agent": self._current_agent,
            "tokens": self._token_count,
            "elapsed_ms": elapsed * 1000,
            "tokens_per_second": tokens_per_second,
            "text": full_text[:200],  # Keep preview only
        })
        logger.info(f"✅ LLM stream complete: {self._token_count} tokens "
                     f"in {elapsed:.2f}s ({tokens_per_second:.0f} tok/s)")

    def set_agent(self, name: str):
        """Set the current agent name for attribution."""
        self._current_agent = name

    @property
    def total_tokens(self) -> int:
        return self._token_count

    @property
    def token_rate_history(self) -> List[Dict[str, Any]]:
        return self._token_history


class AgentAwareStreamingHandler(TokenStreamingHandler):
    """
    Extended streaming handler that tracks which agent generates which tokens.

    Provides richer metadata for multi-agent streaming UIs.
    """

    def __init__(self, on_token: Optional[Callable] = None, on_agent_change: Optional[Callable] = None):
        super().__init__(on_token=on_token)
        self.on_agent_change = on_agent_change
        self._agent_token_counts: Dict[str, int] = {}
        self._agent_start_times: Dict[str, float] = {}

    def set_agent(self, name: str):
        if name != self._current_agent:
            if self.on_agent_change:
                try:
                    self.on_agent_change(self._current_agent, name)
                except Exception:
                    pass
            self._agent_start_times[name] = time.time()
        super().set_agent(name)

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self._agent_token_counts[self._current_agent] = \
            self._agent_token_counts.get(self._current_agent, 0) + 1
        super().on_llm_new_token(token, **kwargs)

    def get_agent_metrics(self) -> Dict[str, Dict]:
        """Get per-agent token and timing metrics."""
        metrics = {}
        for agent, tokens in self._agent_token_counts.items():
            start = self._agent_start_times.get(agent, 0)
            elapsed = time.time() - start if start else 0
            metrics[agent] = {
                "tokens": tokens,
                "elapsed_seconds": round(elapsed, 2),
                "tokens_per_second": round(tokens / max(0.001, elapsed), 1),
            }
        return metrics


# ═══════════════════════════════════════════════════════════════════════════
# 3. Streaming Progress Manager
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StreamingProgress:
    """Tracks progress of a multi-agent streaming session."""
    total_agents: int = 0
    completed_agents: int = 0
    current_agent: str = ""
    total_tokens: int = 0
    events: List[StreamEvent] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: Optional[float] = None

    def add_event(self, event: StreamEvent) -> None:
        self.events.append(event)
        if event.event_type == StreamEventType.AGENT_END:
            self.completed_agents += 1

    def get_progress_pct(self) -> float:
        """Get completion percentage."""
        if self.total_agents == 0:
            return 0.0
        return min(100.0, (self.completed_agents / self.total_agents) * 100)

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    def to_summary(self) -> Dict[str, Any]:
        """Generate a human-readable summary."""
        return {
            "agents_completed": f"{self.completed_agents}/{self.total_agents}",
            "progress_pct": f"{self.get_progress_pct():.0f}%",
            "total_tokens": self.total_tokens,
            "elapsed_seconds": round(self.get_elapsed(), 2),
            "current_agent": self.current_agent,
            "errors": self.errors,
        }


class StreamingProgressManager:
    """
    Manages streaming progress across the entire multi-agent graph execution.

    Usage in graph.py:
        progress = StreamingProgressManager(on_event=my_callback)
        config = {"callbacks": [progress.handler], "configurable": {...}}
        for event in graph.stream(input, config, stream_mode="updates"):
            progress.update_from_graph_event(event)
    """

    def __init__(
        self,
        on_event: Optional[Callable[[StreamEvent], None]] = None,
        on_token: Optional[Callable[[str, Dict], None]] = None,
        known_agents: Optional[List[str]] = None,
    ):
        self.handler = AgentAwareStreamingHandler(on_token=on_token)
        self.on_event = on_event
        self.progress = StreamingProgress(
            total_agents=len(known_agents) if known_agents else 0,
            start_time=time.time(),
        )
        self._known_agents = known_agents or []

    def update_from_graph_event(self, event: Dict[str, Any]) -> StreamEvent:
        """
        Parse a LangGraph stream event and update progress.

        Args:
            event: Dict from graph.stream() output.

        Returns:
            The created StreamEvent.
        """
        stream_event = None

        for node_name, node_output in event.items():
            # Detect agent node execution
            if "assistant" in node_name or "Assistant" in str(node_output):
                if node_name not in self._known_agents:
                    self._known_agents.append(node_name)
                    self.progress.total_agents = len(self._known_agents)

                stream_event = StreamEvent(
                    event_type=StreamEventType.AGENT_START,
                    agent_name=node_name,
                )
                self.progress.current_agent = node_name
                self.handler.set_agent(node_name)

            # Detect tool execution
            elif "tools" in node_name:
                stream_event = StreamEvent(
                    event_type=StreamEventType.TOOL_START,
                    agent_name=node_name,
                    data=node_output,
                )

            # Detect message output
            elif "messages" in node_output:
                messages = node_output.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        stream_event = StreamEvent(
                            event_type=StreamEventType.AGENT_END,
                            agent_name=node_name,
                            data=last_msg.content[:200],
                        )

        if stream_event:
            self.progress.add_event(stream_event)
            if self.on_event:
                try:
                    self.on_event(stream_event)
                except Exception:
                    pass

        return stream_event

    def finalize(self):
        """Mark streaming as complete."""
        self.progress.end_time = time.time()
        final_event = StreamEvent(
            event_type=StreamEventType.COMPLETE,
            data=self.progress.to_summary(),
        )
        self.progress.add_event(final_event)
        if self.on_event:
            try:
                self.on_event(final_event)
            except Exception:
                pass

    def get_summary(self) -> Dict[str, Any]:
        return {
            "progress": self.progress.to_summary(),
            "agent_metrics": self.handler.get_agent_metrics(),
            "total_tokens": self.handler.total_tokens,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Async Streaming Utilities
# ═══════════════════════════════════════════════════════════════════════════

class AsyncStreamGenerator:
    """
    Async generator wrapper for LangGraph streaming.

    Converts synchronous LangGraph stream output into an async iterator
    of StreamEvent objects, suitable for SSE (Server-Sent Events) or
    WebSocket endpoints.
    """

    def __init__(self, graph, input_data: Dict, config: Dict):
        self.graph = graph
        self.input_data = input_data
        self.config = config

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        """
        Yield StreamEvents from the graph execution asynchronously.

        Usage in FastAPI:
            @app.get("/stream")
            async def stream():
                generator = AsyncStreamGenerator(graph, input, config)
                return EventSourceResponse(generator.stream_events())
        """
        try:
            events = await asyncio.to_thread(
                lambda: list(self.graph.stream(self.input_data, self.config, stream_mode="updates"))
            )
        except Exception as exc:
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                data=str(exc),
                metadata={"source": "graph_execution"},
            )
            return

        for event in events:
            for node_name, node_output in event.items():
                if "messages" in node_output:
                    messages = node_output.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if isinstance(last_msg, AIMessage) and last_msg.content:
                            yield StreamEvent(
                                event_type=StreamEventType.TOKEN,
                                data=last_msg.content,
                                agent_name=node_name,
                            )
                else:
                    yield StreamEvent(
                        event_type=StreamEventType.PROGRESS,
                        agent_name=node_name,
                        data=node_output,
                    )
            # Yield control to the event loop
            await asyncio.sleep(0)

        yield StreamEvent(event_type=StreamEventType.COMPLETE)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Token Usage & Cost Calculator
# ═══════════════════════════════════════════════════════════════════════════

class TokenUsageTracker:
    """
    Tracks token usage and estimates cost across agent executions.

    Supports OpenAI, Anthropic Claude, and DeepSeek pricing models.
    """

    # Approximate pricing per 1M tokens (input / output)
    PRICING = {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-3.5-turbo": (0.50, 1.50),
        "claude-3.5-sonnet": (3.00, 15.00),
        "claude-3-haiku": (0.25, 1.25),
        "deepseek-chat": (0.14, 0.28),
        "deepseek-v3": (0.27, 1.10),
        "text-embedding-3-small": (0.02, 0.02),
    }

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._agent_usage: Dict[str, Dict[str, int]] = {}
        self._session_count: int = 0

    def track(self, input_tokens: int, output_tokens: int, agent_name: str = "unknown") -> None:
        """Track token usage for an LLM call."""
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        if agent_name not in self._agent_usage:
            self._agent_usage[agent_name] = {"input": 0, "output": 0, "calls": 0}
        self._agent_usage[agent_name]["input"] += input_tokens
        self._agent_usage[agent_name]["output"] += output_tokens
        self._agent_usage[agent_name]["calls"] += 1
        self._session_count += 1

    def estimate_cost(self, model_name: Optional[str] = None) -> Dict[str, float]:
        """Estimate total cost based on token usage."""
        name = model_name or self.model_name
        input_price, output_price = self.PRICING.get(name, (0.0, 0.0))
        input_cost = (self._input_tokens / 1_000_000) * input_price
        output_cost = (self._output_tokens / 1_000_000) * output_price
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "total_tokens": self._input_tokens + self._output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(input_cost + output_cost, 6),
            "model": name,
        }

    def get_agent_breakdown(self) -> Dict[str, Dict]:
        """Get per-agent token usage breakdown."""
        breakdown = {}
        for agent, usage in self._agent_usage.items():
            total = usage["input"] + usage["output"]
            pct = (total / max(1, self._input_tokens + self._output_tokens)) * 100
            breakdown[agent] = {**usage, "total": total, "percentage": round(pct, 1)}
        return breakdown

    def reset(self) -> None:
        """Reset all counters."""
        self._input_tokens = 0
        self._output_tokens = 0
        self._agent_usage = {}
        self._session_count = 0

    @property
    def total_tokens(self) -> int:
        return self._input_tokens + self._output_tokens


# ═══════════════════════════════════════════════════════════════════════════
# 6. Streaming Graph Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class StreamingGraphWrapper:
    """
    Wraps a compiled LangGraph graph with streaming capabilities.

    Provides:
    - Rich event streaming with agent awareness
    - Token usage tracking
    - Progress reporting
    - Graceful error handling

    Usage:
        wrapper = StreamingGraphWrapper(multi_agentic_graph)
        async for event in wrapper.astream(input_data, config):
            yield event.to_dict()
    """

    def __init__(
        self,
        graph,
        on_token: Optional[Callable] = None,
        on_event: Optional[Callable] = None,
        model_name: str = "gpt-4o-mini",
    ):
        self.graph = graph
        self.on_token = on_token
        self.on_event = on_event
        self.usage_tracker = TokenUsageTracker(model_name=model_name)

    def stream_with_events(
        self,
        input_data: Dict,
        config: Dict,
        known_agents: Optional[List[str]] = None,
    ):
        """
        Synchronous streaming generator yielding StreamEvent objects.

        Args:
            input_data: Graph input (e.g., {"messages": [...]})
            config: RunnableConfig with thread_id etc.
            known_agents: Optional list of agent node names for progress tracking.

        Yields:
            StreamEvent objects.
        """
        progress = StreamingProgressManager(
            on_event=self.on_event,
            on_token=self.on_token,
            known_agents=known_agents,
        )

        try:
            for event in self.graph.stream(input_data, config, stream_mode="updates"):
                stream_event = progress.update_from_graph_event(event)
                if stream_event:
                    yield stream_event

                # Check for interrupts (Human-in-the-Loop)
                snapshot = self.graph.get_state(config)
                if snapshot.next:
                    yield StreamEvent(
                        event_type=StreamEventType.PROGRESS,
                        agent_name="human_loop",
                        data={"message": "Waiting for human approval", "interrupt": True},
                        metadata={"next_nodes": list(snapshot.next)},
                    )

            progress.finalize()

        except Exception as exc:
            logger.error(f"Streaming error: {exc}")
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                data=str(exc),
            )
        finally:
            yield StreamEvent(
                event_type=StreamEventType.COMPLETE,
                data={
                    "summary": progress.get_summary(),
                    "cost": self.usage_tracker.estimate_cost(),
                },
            )

    async def astream_with_events(
        self,
        input_data: Dict,
        config: Dict,
        known_agents: Optional[List[str]] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Async version of stream_with_events."""
        import asyncio
        loop = asyncio.get_event_loop()
        for event in await loop.run_in_executor(
            None,
            lambda: list(self.stream_with_events(input_data, config, known_agents)),
        ):
            yield event

    def invoke_with_cost(
        self,
        input_data: Dict,
        config: Dict,
    ) -> Dict[str, Any]:
        """
        Invoke the graph and return result with cost estimate.

        Returns: {"result": graph_output, "cost": cost_estimate}
        """
        result = self.graph.invoke(input_data, config)
        cost = self.usage_tracker.estimate_cost()
        return {"result": result, "cost": cost}


# Singleton instances for reuse
default_token_handler = TokenStreamingHandler()
default_usage_tracker = TokenUsageTracker()
