"""Chat service — handles streaming and processing for the Health Assistant."""

import asyncio
from typing import AsyncGenerator, Dict, Any
from langchain_core.messages import HumanMessage

from customer_support_chat.app.graph import multi_agentic_graph
from customer_support_chat.app.core.logger import logger


class HealthChatService:
    """
    Service to handle chat interactions with the Health Assistant graph.

    Supports:
    - Synchronous invocation
    - Async streaming with token-level events
    - Thread-based conversation persistence (via LangGraph checkpointer)
    """

    def __init__(self):
        self.graph = multi_agentic_graph

    async def process_message(
        self,
        user_id: str,
        message: str,
        thread_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Process a user message through the multi-agent graph.

        Args:
            user_id: Unique user identifier.
            message: User's text message.
            thread_id: Conversation thread ID for persistence.

        Returns:
            Dict with 'messages' and 'status'.
        """
        logger.info(f"📨 Processing message for user={user_id}, thread={thread_id}")
        logger.info(f"   Message: {message[:100]}...")

        config = {"configurable": {"thread_id": thread_id}}
        input_state = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
        }

        try:
            result = await self.graph.ainvoke(input_state, config=config)
            response_messages = [
                msg for msg in result.get("messages", [])
                if msg.type not in ("tool", "system") and hasattr(msg, "content")
            ]
            if response_messages:
                last_response = response_messages[-1].content
            else:
                last_response = "I processed your request but couldn't generate a response. Please try again."

            logger.info(f"✅ Response generated ({len(last_response)} chars)")
            return {
                "status": "success",
                "response": last_response,
                "messages": result.get("messages", []),
                "thread_id": thread_id,
            }
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
            return {
                "status": "error",
                "response": f"I encountered an error processing your health question: {str(e)}. Please try again.",
                "error": str(e),
            }

    async def stream_response(
        self,
        user_id: str,
        message: str,
        thread_id: str = "default",
    ) -> AsyncGenerator[str, None]:
        """
        Stream the agent's response token by token.

        Args:
            user_id: User identifier.
            message: User message text.
            thread_id: Conversation thread ID.

        Yields:
            Token strings as they are generated.
        """
        config = {"configurable": {"thread_id": thread_id}}
        input_state = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
        }

        try:
            async for event in self.graph.astream_events(input_state, config=config, version="v2"):
                kind = event.get("event", "")
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        yield content
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    yield f"\n🔧 [{tool_name}] "
                elif kind == "on_tool_end":
                    yield ""
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"\nError: {str(e)}"


# Singleton
health_chat_service = HealthChatService()
