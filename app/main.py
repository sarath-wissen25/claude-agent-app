from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

from mlflow.genai.agent_server import AgentServer, invoke, setup_mlflow_git_based_version_tracking
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse, create_text_output_item

from app.gateway_agent import GatewayAgent
from app.memory import build_session_store
from app.schemas import ChatMessage


store = build_session_store()
agent = GatewayAgent(store=store)
agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=True)
app = agent_server.app
setup_mlflow_git_based_version_tracking()


@invoke()
async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    payload = _request_to_dict(request)
    messages = _extract_messages(payload)
    conversation_id = _extract_conversation_id(payload) or str(uuid.uuid4())
    custom_inputs = _custom_inputs(payload)

    result = await agent.run(
        messages=messages,
        conversation_id=conversation_id,
        user_id=_string_or_none(custom_inputs.get("user_id") or payload.get("user_id")),
        metadata=_dict_or_empty(custom_inputs.get("metadata") or payload.get("metadata")),
    )
    return _responses_payload(result)


def _request_to_dict(request: ResponsesAgentRequest | dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(request, dict):
        return request
    if hasattr(request, "model_dump"):
        return request.model_dump(exclude_none=True)
    return dict(request)


def _extract_messages(request: dict[str, Any]) -> list[ChatMessage]:
    raw_messages = request.get("messages") or request.get("input") or []
    if isinstance(raw_messages, str):
        raw_messages = [{"role": "user", "content": raw_messages}]
    return [ChatMessage(**message) for message in raw_messages]


def _extract_conversation_id(request: dict[str, Any]) -> str | None:
    for name in ("conversation_id", "session_id", "thread_id"):
        if request.get(name):
            return str(request[name])
    custom_inputs = _custom_inputs(request)
    for name in ("conversation_id", "session_id", "thread_id"):
        if custom_inputs.get(name):
            return str(custom_inputs[name])
    return None


def _custom_inputs(request: dict[str, Any]) -> dict[str, Any]:
    value = request.get("custom_inputs") or {}
    return value if isinstance(value, dict) else {}


def _responses_payload(result: dict[str, Any]) -> ResponsesAgentResponse:
    return ResponsesAgentResponse(
        output=[create_text_output_item(text=result["output"])],
        custom_outputs={
            "conversation_id": result["conversation_id"],
            "session_id": result.get("session_id"),
            **result.get("metadata", {}),
        },
    )


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_or_none(value: Any) -> str | None:
    return str(value) if value else None


def main() -> None:
    agent_server.run(app_import_string="app.main:app")


if __name__ == "__main__":
    main()


class DatabricksAppsGatewayAgent(ResponsesAgent):
    def predict(self, request: Any) -> dict[str, Any]:
        response = asyncio.run(non_streaming(request))
        return response.model_dump(exclude_none=True)
