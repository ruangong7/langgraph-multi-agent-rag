"""FastAPI web application entry point for the Personal Health Assistant."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from customer_support_chat.app.services.chat_service import health_chat_service
from customer_support_chat.app.core.logger import logger

app = FastAPI(
    title="Personal Health Assistant",
    description="Multi-agent LangGraph health assistant with Knowledge Graph and RAG",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_id: str
    message: str
    thread_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    status: str
    response: str
    thread_id: str


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "Personal Health Assistant", "version": "2.0.0"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a health-related chat message through the multi-agent system."""
    try:
        result = await health_chat_service.process_message(
            user_id=request.user_id,
            message=request.message,
            thread_id=request.thread_id,
        )
        return ChatResponse(
            status=result["status"],
            response=result["response"],
            thread_id=result.get("thread_id", request.thread_id),
        )
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """API root."""
    return {
        "message": "🏥 Personal Health Assistant API",
        "docs": "/docs",
        "endpoints": {
            "POST /chat": "Send a health query",
            "GET /health": "Health check",
        },
    }
