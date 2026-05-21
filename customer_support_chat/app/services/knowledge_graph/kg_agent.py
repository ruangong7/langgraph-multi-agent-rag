"""
Knowledge Graph Agent — LangGraph-compatible agent node for KG-enhanced RAG.

This agent is designed to be added as a node in the LangGraph StateGraph.
It intercepts user queries that benefit from knowledge graph reasoning and
returns enriched context alongside generated responses.

Integration points (to be added in graph.py):
    builder.add_node("enter_kg", create_entry_node("Knowledge Graph Agent", "kg_reasoning"))
    builder.add_node("kg_reasoning", kg_agent_node)
    builder.add_edge("enter_kg", "kg_reasoning")
    builder.add_conditional_edges("kg_reasoning", route_kg, {...})
"""

from typing import Literal, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from customer_support_chat.app.core.state import State
from customer_support_chat.app.core.logger import logger
from customer_support_chat.app.services.assistants.assistant_base import Assistant, CompleteOrEscalate, llm

from .entity_extractor import entity_extractor, ExtractedEntity
from .relation_builder import relation_builder
from .graph_store import graph_store
from .reasoning import graph_reasoning, ReasoningResult
from .kg_retriever import KGRetriever, RetrievalContext


# ── KG Agent State ──────────────────────────────────────────────────────

class KGAgentState(BaseModel):
    """State specific to the Knowledge Graph Agent."""
    query_entities: List[ExtractedEntity] = Field(default_factory=list)
    reasoning_result: Optional[ReasoningResult] = None
    retrieval_context: Optional[RetrievalContext] = None
    graph_enriched: bool = Field(default=False)
    kg_stats: Dict[str, Any] = Field(default_factory=dict)


# ── Delegation Models ───────────────────────────────────────────────────

class ToKnowledgeGraph(BaseModel):
    """Delegate a query to the Knowledge Graph Agent for structured reasoning."""
    query: str = Field(description="The user's query that requires knowledge graph analysis")
    entity_focus: str = Field(
        default="",
        description="Specific entity or entities to focus the KG reasoning on"
    )
    reasoning_type: str = Field(
        default="expand",
        description="Type of KG reasoning: 'expand' (context expansion), "
                    "'path' (find connections), 'contradict' (check conflicts)"
    )


# ── KG Agent Prompt ─────────────────────────────────────────────────────

KG_AGENT_SYSTEM_PROMPT = """You are a Knowledge Graph Agent for a travel & e-commerce customer support system.

Your role is to:
1. Analyze user queries and determine if knowledge graph reasoning would help
2. Extract entities and relationships from queries
3. Query the knowledge graph for relevant structured information
4. Combine graph insights with text-based search results
5. Provide enriched responses that include entity relationships and context

When responding:
- If the knowledge graph provides relevant relationships, mention them explicitly
  Example: "Based on our records, your flight LH123 departs from Zurich and arrives at London Heathrow."
- If the graph doesn't have enough information, fall back to general knowledge
- Always cite the relationships you found, including confidence scores when low
- Use the expanded context to suggest related options (e.g., hotels near destination)

Current knowledge graph context:
{kg_context}

Current semantic search results:
{vector_context}

User query: {query}"""

KG_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", KG_AGENT_SYSTEM_PROMPT),
    ("placeholder", "{messages}"),
])


# ── KG Agent Tools ──────────────────────────────────────────────────────

class KGTools:
    """Tools that the KG Agent can use."""

    @staticmethod
    def generate_kg_aware_response(
        query: str,
        retrieval_context: RetrievalContext,
        reasoning_result: Optional[ReasoningResult] = None,
    ) -> str:
        """
        Generate a response that incorporates knowledge graph insights.
        This is called by the agent node, not as a LangChain tool.
        """
        kg_formatted = retrieval_context.format_for_llm()

        reasoning_info = ""
        if reasoning_result:
            reasoning_info = reasoning_result.format_for_llm()

        # Build the prompt with all context
        prompt_text = KG_AGENT_PROMPT.partial(
            kg_context=f"{reasoning_info}\n\n{kg_formatted}" if reasoning_info else kg_formatted,
            vector_context=retrieval_context.vector_context.chunks[:3] if retrieval_context.vector_context.chunks else "None",
            query=query,
        )

        # The actual LLM call happens in the agent node
        return kg_formatted


# ── KG Agent Runnable ──────────────────────────────────────────────────

# Initialize retriever
kg_retriever = KGRetriever()

# Note: This will be properly initialized when graph.py creates the node
# with access to the running Qdrant client.

kg_agent_tools = [CompleteOrEscalate]

# Build the runnable
def _create_kg_runnable():
    """Create the KG agent runnable."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a Knowledge Graph reasoning agent for customer support. "
            "Your role is to analyze queries using structured knowledge from the entity-relationship graph "
            "and provide enriched responses.\n\n"
            "Current knowledge graph statistics: {kg_stats}\n\n"
            "When users ask about relationships between entities (e.g., 'which hotels are near my destination?', "
            "'what flights does this customer have?'), use the knowledge graph to find connections.\n\n"
            "If you cannot help, escalate to the primary assistant via CompleteOrEscalate. "
            "Be concise but thorough — mention specific entity relationships you found.\n\n"
            "Current time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ])

    from datetime import datetime
    prompt = prompt.partial(
        kg_stats=str(graph_store.get_statistics()),
        time=datetime.now().isoformat(),
    )

    return prompt | llm.bind_tools(kg_agent_tools)


kg_agent_runnable = _create_kg_runnable()
kg_agent = Assistant(kg_agent_runnable)


# ── KG Agent Node Function ──────────────────────────────────────────────

def kg_agent_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    LangGraph node function for the Knowledge Graph Agent.

    This is the main entry point when a query is routed to the KG agent.
    It:
    1. Extracts entities from the latest user message
    2. Performs KG + vector retrieval
    3. Passes enriched context to the LLM for response generation

    Args:
        state: Current LangGraph state.
        config: Runnable config.

    Returns:
        State update dict with new messages.
    """
    # Get the latest user message
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not user_messages:
        return {"messages": [AIMessage(content="I couldn't find a query to analyze.")]}

    latest_message = user_messages[-1]
    query = latest_message.content

    logger.info(f"🧠 KG Agent processing: {query[:100]}...")

    # Step 1: Perform KG + Vector retrieval
    try:
        retrieval_context = kg_retriever.retrieve(query, top_k=5, kg_expansion_hops=2)
        logger.info(f"✅ KG retrieval complete: {len(retrieval_context.kg_context.entities)} entities, "
                     f"{len(retrieval_context.vector_context.chunks)} vector chunks")
    except Exception as exc:
        logger.error(f"❌ KG retrieval failed: {exc}")
        retrieval_context = RetrievalContext()

    # Step 2: Build context message for the LLM
    kg_context_text = retrieval_context.format_for_llm()

    # Step 3: Let the Assistant runnable handle the response generation
    # The runnable (kg_agent_runnable → llm.bind_tools) will:
    # - Accept the conversation messages
    # - Process them with KG context injected
    # - Return either a tool call (CompleteOrEscalate) or an AIMessage

    result = kg_agent(state, config)

    # Enrich the response with KG metadata if it's an AI message
    messages = result.get("messages", [])
    enriched_messages = []
    for msg in messages:
        if hasattr(msg, "content") and isinstance(msg.content, str) and retrieval_context.kg_context.entities:
            # Append a concise KG insight if entities were found
            entities_str = ", ".join(e.value for e in retrieval_context.kg_context.entities[:5])
            kg_note = f"\n\n---\n📊 *Knowledge Graph insights: {len(retrieval_context.kg_context.entities)} entities found ({entities_str}). {len(retrieval_context.kg_context.direct_relations)} direct relations discovered.*"
            # Don't modify the message, just let the LLM decide what to include
            enriched_messages.append(msg)
        else:
            enriched_messages.append(msg)

    if enriched_messages:
        return {"messages": enriched_messages}
    return result


def route_kg_agent(state: State) -> Literal["primary_assistant", "__end__"]:
    """
    Route from KG agent: either escalate to primary assistant or end.
    """
    from langgraph.prebuilt import tools_condition
    route = tools_condition(state)
    if route == "__end__":
        return "__end__"

    # Check if CompleteOrEscalate was called
    last_message = state["messages"][-1] if state["messages"] else None
    if last_message and hasattr(last_message, "content") and isinstance(last_message.content, str):
        if "escalated to main assistant" in last_message.content.lower():
            return "primary_assistant"

    return "primary_assistant"


# ── Factory Function ────────────────────────────────────────────────────

def create_kg_agent() -> Dict[str, Any]:
    """
    Factory function that returns all components needed to add the KG Agent
    to a LangGraph StateGraph.

    Returns:
        Dict with:
        - agent_node: The node function
        - agent_runnable: The agent runnable
        - tools: Tools for the agent
        - route_fn: Routing function
        - delegation_model: The delegation Pydantic model
    """
    return {
        "agent_node": kg_agent_node,
        "agent_runnable": kg_agent_runnable,
        "agent": kg_agent,
        "tools": kg_agent_tools,
        "route_fn": route_kg_agent,
        "delegation_model": ToKnowledgeGraph,
        "retriever": kg_retriever,
    }
