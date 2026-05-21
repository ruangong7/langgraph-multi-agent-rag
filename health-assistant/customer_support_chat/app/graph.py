"""
Personal Health Assistant — LangGraph StateGraph with 9 specialized agents.

Architecture (Supervisor Pattern):
  User → Guardrails → Primary Health Assistant → Specialized Agent → Tools → Back

Agents:
  1. Appointment    — Book/cancel medical appointments
  2. Medication     — Manage medications, reminders, interactions
  3. Emergency      — First aid and emergency guidance
  4. Health Tips    — Exercise, diet, sleep, mental health
  5. Medical Record — View/add medical history
  6. Health Assess  — Symptom checker, risk evaluation
  7. Medical KB     — Medical literature and drug search
  8. Medical KG     — Disease-Symptom-Medication knowledge graph
  9. Primary        — Orchestrator, routes to specialists
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

from customer_support_chat.app.core.state import State
from customer_support_chat.app.core.logger import logger
from customer_support_chat.app.services.utils import (
    create_tool_node_with_fallback,
    create_entry_node,
)

# ── Guardrails ────────────────────────────────────────────────────────

from customer_support_chat.app.services.guardrails.guardrail_agents import (
    jailbreak_guardrail_agent,
    jailbreak_guardrail_agent_instructions,
    relevance_guardrail_agent,
    relevance_guardrail_agent_instructions,
)

# ── Base ──────────────────────────────────────────────────────────────

from customer_support_chat.app.services.assistants.assistant_base import (
    Assistant, CompleteOrEscalate, llm,
)

# ── Primary Assistant ─────────────────────────────────────────────────

from customer_support_chat.app.services.assistants.primary_assistant import (
    primary_assistant, primary_assistant_tools,
    ToAppointmentBooking, ToMedicationManagement, ToEmergencyAssist,
    ToHealthTips, ToMedicalRecords, ToHealthAssessment,
    ToMedicalKnowledgeSearch, ToMedicalKnowledgeGraph,
)

# ── Specialized Agents ────────────────────────────────────────────────

from customer_support_chat.app.services.assistants.appointment_assistant import (
    appointment_booking_assistant,
    appointment_safe_tools,
    appointment_sensitive_tools,
)

from customer_support_chat.app.services.assistants.medication_assistant import (
    medication_management_assistant,
    medication_safe_tools,
    medication_sensitive_tools,
)

from customer_support_chat.app.services.assistants.emergency_assistant import (
    emergency_assistant,
    emergency_safe_tools,
)

from customer_support_chat.app.services.assistants.health_tips_assistant import (
    health_tips_assistant,
    health_tips_safe_tools,
)

from customer_support_chat.app.services.assistants.medical_record_assistant import (
    medical_record_assistant,
    medical_record_safe_tools,
    medical_record_sensitive_tools,
)

from customer_support_chat.app.services.assistants.health_assessment_assistant import (
    health_assessment_assistant,
    health_assessment_safe_tools,
)

from customer_support_chat.app.services.assistants.medical_kb_assistant import (
    medical_kb_agent,
    medical_kb_safe_tools,
)

from customer_support_chat.app.services.knowledge_graph.medical_kg_agent import (
    kg_agent_node, kg_agent_tools, ToMedicalKnowledgeGraph,
)


# ═══════════════════════════════════════════════════════════════════════
# Initialize the StateGraph
# ═══════════════════════════════════════════════════════════════════════

builder = StateGraph(State)


# ── User Info Node ────────────────────────────────────────────────────

def user_info(state: State, config: RunnableConfig):
    """Fetch user health profile for context."""
    from customer_support_chat.app.services.tools.health import fetch_user_health_profile
    user_id = state.get("user_id", "default_user")
    profile_str = fetch_user_health_profile.invoke({"user_id": user_id})
    return {"user_info": profile_str}

builder.add_node("fetch_user_info", user_info)


# ── Guardrail Node ────────────────────────────────────────────────────

def guardrail_check(state: State, config: RunnableConfig):
    """Check user input for safety and medical relevance."""
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not user_messages:
        logger.warning("No user message for guardrail check. Allowing.")
        return {"messages": [HumanMessage(content="No input detected. Please share your health concern.")]}

    latest_user_message = user_messages[-1]
    user_input = latest_user_message.content

    logger.info(f"🛡️ Guardrail check: '{user_input[:80]}...'")

    # 1. Jailbreak detection
    jailbreak_prompt = f"{jailbreak_guardrail_agent_instructions}\n\nUser Input: {user_input}"
    jailbreak_result = jailbreak_guardrail_agent.invoke(jailbreak_prompt)

    if not jailbreak_result.is_safe:
        logger.warning(f"🚨 Jailbreak detected: {jailbreak_result.reasoning}")
        return {"messages": [HumanMessage(
            content=f"I cannot process that request as it appears to violate safety guidelines. "
                    f"If you have a health concern, I'd be happy to help."
        )]}

    # 2. Relevance check
    relevance_prompt = f"{relevance_guardrail_agent_instructions}\n\nUser Input: {user_input}"
    relevance_result = relevance_guardrail_agent.invoke(relevance_prompt)

    if not relevance_result.is_relevant:
        logger.warning(f"⚠️ Irrelevant input: {relevance_result.reasoning}")
        return {"messages": [HumanMessage(
            content=f"I'm a Personal Health Assistant — I can help with health and medical questions. "
                    f"Your query seems to be about something else ({relevance_result.category}). "
                    f"Is there a health-related question I can assist with?"
        )]}

    logger.info(f"✅ Passed guardrails. Category: {relevance_result.category}")
    return {"messages": []}

builder.add_node("guardrail_check", guardrail_check)


# ═══════════════════════════════════════════════════════════════════════
# Helper: route back to primary or stay
# ═══════════════════════════════════════════════════════════════════════

def should_route_to_primary(state: State) -> bool:
    if state.get("messages") and len(state["messages"]) > 0:
        last_message = state["messages"][-1]
        if hasattr(last_message, 'content') and isinstance(last_message.content, str):
            return 'Task completed/escalated to main assistant' in last_message.content
    return False


# ═══════════════════════════════════════════════════════════════════════
# Appointment Agent
# ═══════════════════════════════════════════════════════════════════════

builder.add_node("enter_appointment",
    create_entry_node("Appointment Booking Assistant", "book_appointment"))
builder.add_node("appointment_agent", appointment_booking_assistant)
builder.add_edge("enter_appointment", "appointment_agent")
builder.add_node("appointment_safe_tools",
    create_tool_node_with_fallback(appointment_safe_tools))
builder.add_node("appointment_sensitive_tools",
    create_tool_node_with_fallback(appointment_sensitive_tools))

def route_appointment(state: State) -> Literal[
    "appointment_safe_tools", "appointment_sensitive_tools", "primary_assistant", "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    safe_names = {t.name for t in appointment_safe_tools}
    if all(tc["name"] in safe_names for tc in tool_calls):
        return "appointment_safe_tools"
    return "appointment_sensitive_tools"

def route_appointment_tools(state: State) -> Literal["appointment_agent", "primary_assistant"]:
    return "primary_assistant" if should_route_to_primary(state) else "appointment_agent"

builder.add_conditional_edges("appointment", lambda s: route_appointment(s))
builder.add_conditional_edges("appointment_safe_tools", route_appointment_tools)
builder.add_conditional_edges("appointment_sensitive_tools", route_appointment_tools)


# ═══════════════════════════════════════════════════════════════════════
# Medication Agent
# ═══════════════════════════════════════════════════════════════════════

builder.add_node("enter_medication",
    create_entry_node("Medication Management Assistant", "manage_medication"))
builder.add_node("medication_agent", medication_management_assistant)
builder.add_edge("enter_medication", "medication_agent")
builder.add_node("medication_safe_tools",
    create_tool_node_with_fallback(medication_safe_tools))
builder.add_node("medication_sensitive_tools",
    create_tool_node_with_fallback(medication_sensitive_tools))

def route_medication(state: State) -> Literal[
    "medication_safe_tools", "medication_sensitive_tools", "primary_assistant", "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    safe_names = {t.name for t in medication_safe_tools}
    if all(tc["name"] in safe_names for tc in tool_calls):
        return "medication_safe_tools"
    return "medication_sensitive_tools"

def route_medication_tools(state: State) -> Literal["medication_agent", "primary_assistant"]:
    return "primary_assistant" if should_route_to_primary(state) else "medication_agent"

builder.add_conditional_edges("medication_agent", route_medication)
builder.add_conditional_edges("medication_safe_tools", route_medication_tools)
builder.add_conditional_edges("medication_sensitive_tools", route_medication_tools)


# ═══════════════════════════════════════════════════════════════════════
# Emergency Agent
# ═══════════════════════════════════════════════════════════════════════

builder.add_node("enter_emergency",
    create_entry_node("Emergency Assistant", "emergency_assist"))
builder.add_node("emergency_agent", emergency_assistant)
builder.add_edge("enter_emergency", "emergency_agent")
builder.add_node("emergency_tools",
    create_tool_node_with_fallback(emergency_safe_tools))

def route_emergency(state: State) -> Literal[
    "emergency_tools", "primary_assistant", "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    return "emergency_tools"

def route_emergency_tools(state: State) -> Literal["emergency_agent", "primary_assistant"]:
    return "primary_assistant" if should_route_to_primary(state) else "emergency_agent"

builder.add_conditional_edges("emergency_agent", route_emergency)
builder.add_conditional_edges("emergency_tools", route_emergency_tools)


# ═══════════════════════════════════════════════════════════════════════
# Health Tips Agent
# ═══════════════════════════════════════════════════════════════════════

builder.add_node("enter_health_tips",
    create_entry_node("Health & Wellness Advisor", "health_tips"))
builder.add_node("health_tips_agent", health_tips_assistant)
builder.add_edge("enter_health_tips", "health_tips_agent")
builder.add_node("health_tips_tools",
    create_tool_node_with_fallback(health_tips_safe_tools))

def route_health_tips(state: State) -> Literal[
    "health_tips_tools", "primary_assistant", "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    return "health_tips_tools"

def route_health_tips_tools(state: State) -> Literal["health_tips_agent", "primary_assistant"]:
    return "primary_assistant" if should_route_to_primary(state) else "health_tips_agent"

builder.add_conditional_edges("health_tips_agent", route_health_tips)
builder.add_conditional_edges("health_tips_tools", route_health_tips_tools)


# ═══════════════════════════════════════════════════════════════════════
# Medical Record Agent
# ═══════════════════════════════════════════════════════════════════════

builder.add_node("enter_medical_records",
    create_entry_node("Medical Records Manager", "medical_records"))
builder.add_node("medical_records_agent", medical_record_assistant)
builder.add_edge("enter_medical_records", "medical_records_agent")
builder.add_node("medical_records_safe_tools",
    create_tool_node_with_fallback(medical_record_safe_tools))
builder.add_node("medical_records_sensitive_tools",
    create_tool_node_with_fallback(medical_record_sensitive_tools))

def route_medical_records(state: State) -> Literal[
    "medical_records_safe_tools", "medical_records_sensitive_tools", "primary_assistant", "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    safe_names = {t.name for t in medical_record_safe_tools}
    if all(tc["name"] in safe_names for tc in tool_calls):
        return "medical_records_safe_tools"
    return "medical_records_sensitive_tools"

def route_medical_records_tools(state: State) -> Literal["medical_records_agent", "primary_assistant"]:
    return "primary_assistant" if should_route_to_primary(state) else "medical_records_agent"

builder.add_conditional_edges("medical_records_agent", route_medical_records)
builder.add_conditional_edges("medical_records_safe_tools", route_medical_records_tools)
builder.add_conditional_edges("medical_records_sensitive_tools", route_medical_records_tools)


# ═══════════════════════════════════════════════════════════════════════
# Health Assessment Agent
# ═══════════════════════════════════════════════════════════════════════

builder.add_node("enter_health_assessment",
    create_entry_node("Health Assessment Specialist", "health_assessment"))
builder.add_node("health_assessment_agent", health_assessment_assistant)
builder.add_edge("enter_health_assessment", "health_assessment_agent")
builder.add_node("health_assessment_tools",
    create_tool_node_with_fallback(health_assessment_safe_tools))

def route_health_assessment(state: State) -> Literal[
    "health_assessment_tools", "primary_assistant", "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    return "health_assessment_tools"

def route_health_assessment_tools(state: State) -> Literal["health_assessment_agent", "primary_assistant"]:
    return "primary_assistant" if should_route_to_primary(state) else "health_assessment_agent"

builder.add_conditional_edges("health_assessment_agent", route_health_assessment)
builder.add_conditional_edges("health_assessment_tools", route_health_assessment_tools)


# ═══════════════════════════════════════════════════════════════════════
# Medical Knowledge Search Agent
# ═══════════════════════════════════════════════════════════════════════

builder.add_node("enter_medical_kb",
    create_entry_node("Medical Knowledge Search", "medical_kb"))
builder.add_node("medical_kb_agent_node", medical_kb_agent)
builder.add_edge("enter_medical_kb", "medical_kb_agent_node")
builder.add_node("medical_kb_tools",
    create_tool_node_with_fallback(medical_kb_safe_tools))

def route_medical_kb(state: State) -> Literal[
    "medical_kb_tools", "primary_assistant", "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    return "medical_kb_tools"

def route_medical_kb_tools(state: State) -> Literal["medical_kb_agent_node", "primary_assistant"]:
    return "primary_assistant" if should_route_to_primary(state) else "medical_kb_agent_node"

builder.add_conditional_edges("medical_kb_agent_node", route_medical_kb)
builder.add_conditional_edges("medical_kb_tools", route_medical_kb_tools)


# ═══════════════════════════════════════════════════════════════════════
# Medical Knowledge Graph Agent
# ═══════════════════════════════════════════════════════════════════════

builder.add_node("enter_kg",
    create_entry_node("Medical Knowledge Graph Agent", "medical_kg"))
builder.add_node("kg_reasoning", kg_agent_node)
builder.add_edge("enter_kg", "kg_reasoning")
builder.add_node("kg_tools",
    create_tool_node_with_fallback(kg_agent_tools))

def route_kg(state: State) -> Literal[
    "kg_tools", "primary_assistant", "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    return "kg_tools"

def route_kg_tools(state: State) -> Literal["kg_reasoning", "primary_assistant"]:
    return "primary_assistant" if should_route_to_primary(state) else "kg_reasoning"

builder.add_conditional_edges("kg_reasoning", route_kg)
builder.add_conditional_edges("kg_tools", route_kg_tools)


# ═══════════════════════════════════════════════════════════════════════
# Primary Assistant — Orchestrator
# ═══════════════════════════════════════════════════════════════════════

builder.add_node("primary_assistant", primary_assistant)
builder.add_node("primary_assistant_tools",
    create_tool_node_with_fallback(primary_assistant_tools))
builder.add_edge("fetch_user_info", "primary_assistant")

def route_primary_assistant(state: State) -> Literal[
    "primary_assistant_tools",
    "enter_appointment",
    "enter_medication",
    "enter_emergency",
    "enter_health_tips",
    "enter_medical_records",
    "enter_health_assessment",
    "enter_medical_kb",
    "enter_kg",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        tool_name = tool_calls[0]["name"]
        routing_map = {
            ToAppointmentBooking.__name__: "enter_appointment",
            ToMedicationManagement.__name__: "enter_medication",
            ToEmergencyAssist.__name__: "enter_emergency",
            ToHealthTips.__name__: "enter_health_tips",
            ToMedicalRecords.__name__: "enter_medical_records",
            ToHealthAssessment.__name__: "enter_health_assessment",
            ToMedicalKnowledgeSearch.__name__: "enter_medical_kb",
            ToMedicalKnowledgeGraph.__name__: "enter_kg",
        }
        if tool_name in routing_map:
            logger.info(f"🔀 Routing to: {routing_map[tool_name]}")
            return routing_map[tool_name]
        return "primary_assistant_tools"
    return "primary_assistant"

builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    {
        "enter_appointment": "enter_appointment",
        "enter_medication": "enter_medication",
        "enter_emergency": "enter_emergency",
        "enter_health_tips": "enter_health_tips",
        "enter_medical_records": "enter_medical_records",
        "enter_health_assessment": "enter_health_assessment",
        "enter_medical_kb": "enter_medical_kb",
        "enter_kg": "enter_kg",
        "primary_assistant_tools": "primary_assistant_tools",
        END: END,
    },
)
builder.add_edge("primary_assistant_tools", "primary_assistant")

# ── Entry point routing ───────────────────────────────────────────────

builder.add_edge(START, "fetch_user_info")
builder.add_edge("fetch_user_info", "guardrail_check")
builder.add_edge("guardrail_check", "primary_assistant")


# ═══════════════════════════════════════════════════════════════════════
# Compile with Human-in-the-Loop interrupts
# ═══════════════════════════════════════════════════════════════════════

interrupt_nodes = [
    "appointment_sensitive_tools",
    "medication_sensitive_tools",
    "medical_records_sensitive_tools",
]

memory = MemorySaver()
multi_agentic_graph = builder.compile(
    checkpointer=memory,
    interrupt_before=interrupt_nodes,
)
