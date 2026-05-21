"""Utility functions for the Health Assistant multi-agent system."""

from typing import List, Callable
from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage, AIMessage
from langgraph.prebuilt import ToolNode
from customer_support_chat.app.core.logger import logger


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    """
    Create a node that enters a specialized agent's dialog state.

    Args:
        assistant_name: Display name for logging.
        new_dialog_state: The dialog state key to push onto the stack.

    Returns:
        A callable node function for the StateGraph.
    """
    def entry_node(state):
        logger.info(f"➡️ Entering: {assistant_name} (dialog_state → {new_dialog_state})")
        return {"dialog_state": new_dialog_state}
    return entry_node


def create_tool_node_with_fallback(tools: List) -> ToolNode:
    """
    Create a ToolNode that handles both tool execution and fallback errors.

    Unlike the standard ToolNode, this wraps tool errors gracefully so
    the agent can recover instead of crashing.
    """
    def handle_tool_error(state) -> dict:
        """Handle tool execution errors gracefully."""
        error = state.get("error", "Unknown tool error")
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\nPlease fix your input and try again.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }

    return ToolNode(tools).with_fallbacks(
        [handle_tool_error], exception_key="error"
    )


def health_profile_to_string(profile_data: dict) -> str:
    """
    Format user health profile into a readable string for LLM context.

    Args:
        profile_data: Dict from the database with user profile fields.

    Returns:
        Formatted string.
    """
    if not profile_data:
        return "No health profile available."
    lines = ["## User Health Profile"]
    if "name" in profile_data:
        lines.append(f"- Name: {profile_data['name']}")
    if "date_of_birth" in profile_data:
        lines.append(f"- DOB: {profile_data['date_of_birth']}")
    if "gender" in profile_data:
        lines.append(f"- Gender: {profile_data['gender']}")
    if "blood_type" in profile_data:
        lines.append(f"- Blood Type: {profile_data['blood_type']}")
    if "allergies" in profile_data:
        lines.append(f"- Allergies: {profile_data['allergies']}")
    if "chronic_conditions" in profile_data:
        lines.append(f"- Chronic Conditions: {profile_data['chronic_conditions']}")
    if "emergency_contact" in profile_data:
        lines.append(f"- Emergency Contact: {profile_data['emergency_contact']}")
    return "\n".join(lines)
