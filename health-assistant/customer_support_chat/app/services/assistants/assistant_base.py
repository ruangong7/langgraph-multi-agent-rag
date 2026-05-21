"""
Base Assistant — LangGraph-compatible agent wrapper for the Health Assistant system.

All specialized health agents extend this pattern.
"""

from typing import List, Optional, Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from customer_support_chat.app.core.settings import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME
from customer_support_chat.app.core.logger import logger


# ── Shared LLM instance ───────────────────────────────────────────────

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    temperature=0.1,
)


# ── CompleteOrEscalate tool ───────────────────────────────────────────

class CompleteOrEscalate:
    """
    Call when the specialized agent has completed its task or needs
    to escalate back to the primary health assistant.
    """
    def __init__(self, reason: str = ""):
        self.reason = reason
        self.result = "Task completed/escalated to main assistant"

    def __call__(self) -> str:
        return self.result


class Assistant:
    """
    A LangGraph-compatible reactive agent wrapper.

    Each specialized health agent uses this pattern:
    1. Define delegation tool(s) for routing
    2. Define domain tools for the agent's tasks
    3. Provide a system prompt describing the agent's role
    """

    def __init__(
        self,
        delegation_tools: List[type],
        domain_tools: List[BaseTool],
        agent_name: str,
        system_prompt: str,
        model: Optional[BaseChatModel] = None,
    ):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.model = model or llm
        all_tools = delegation_tools + domain_tools
        self.tools = all_tools
        self.agent = create_react_agent(
            self.model,
            all_tools,
            state_modifier=system_prompt,
        )
        logger.info(f"🤖 Agent created: {agent_name} ({len(all_tools)} tools)")

    def __call__(self, state, config=None):
        """Invoke the agent synchronously."""
        return self.agent.invoke(state, config=config)

    async def ainvoke(self, state, config=None):
        """Invoke the agent asynchronously."""
        return await self.agent.ainvoke(state, config=config)
