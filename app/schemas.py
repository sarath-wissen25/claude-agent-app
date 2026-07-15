from typing import Any, Literal

from pydantic import BaseModel, Field


Role = Literal["system", "user", "assistant", "tool"]
Content = str | list[dict[str, Any]]


class ChatMessage(BaseModel):
    role: Role
    content: Content


class InvocationRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    input: str | list[ChatMessage] | None = None
    conversation_id: str | None = None
    user_id: str | None = None
    stream: bool = False
    custom_inputs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InvocationResponse(BaseModel):
    conversation_id: str
    session_id: str | None
    output: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionSummary(BaseModel):
    conversation_id: str
    claude_session_id: str | None
    turns: int
    last_updated_at: str | None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    sdk_importable: bool
    memory_backend: str
    details: str
