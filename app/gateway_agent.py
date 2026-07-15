from __future__ import annotations

import os
import time
from typing import Any

import mlflow
from openai import AsyncOpenAI

from app.memory import SessionStore, StoredSession
from app.schemas import ChatMessage


DEFAULT_SYSTEM_PROMPT = """You are a pragmatic assistant hosted on Databricks Apps with MLflow Agent Server.
Answer clearly, use the provided short-term conversation memory when it is relevant,
and call out operational assumptions that matter for deployment, monitoring, sessions,
and memory tests."""


class GatewayAgent:
    def __init__(self, store: SessionStore) -> None:
        self.store = store
        self.system_prompt = os.getenv("AGENT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
        self.model = os.getenv("LITELLM_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("LITELLM_TEMPERATURE", "0.2"))
        self.max_tokens = _optional_int("LITELLM_MAX_TOKENS")
        self.short_memory_turns = int(os.getenv("SHORT_MEMORY_MAX_TURNS", "12"))
        self.client = AsyncOpenAI(
            api_key=os.getenv("LITELLM_API_KEY", "not-required"),
            base_url=_base_url(),
            timeout=float(os.getenv("LITELLM_TIMEOUT_SECONDS", "120")),
        )

    @mlflow.trace
    async def run(
        self,
        messages: list[ChatMessage],
        conversation_id: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt = _latest_user_prompt(messages)
        if not prompt:
            raise ValueError("A user message is required.")

        session = self.store.get(conversation_id)
        started_at = time.time()
        gateway_messages = self._gateway_messages(messages, session)
        output, usage = await self._query_gateway(gateway_messages)

        session.turns.append(
            {
                "user_id": user_id,
                "request": [message.model_dump() for message in messages],
                "response": output,
                "metadata": metadata or {},
                "latency_ms": round((time.time() - started_at) * 1000),
                "created_at": time.time(),
            }
        )
        self.store.save(session)

        return {
            "conversation_id": conversation_id,
            "session_id": conversation_id,
            "output": output,
            "metadata": {
                "turn_count": len(session.turns),
                "latency_ms": round((time.time() - started_at) * 1000),
                "model": self.model,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "memory_backend": self.store.backend_name,
            },
        }

    def _gateway_messages(self, messages: list[ChatMessage], session: StoredSession) -> list[dict[str, str]]:
        gateway_messages = [{"role": "system", "content": self.system_prompt}]
        memory = self._short_memory(session)
        if memory:
            gateway_messages.append({"role": "system", "content": memory})
        gateway_messages.extend(
            {"role": _gateway_role(message.role), "content": _content_to_text(message.content)}
            for message in messages
            if _content_to_text(message.content).strip()
        )
        return gateway_messages

    def _short_memory(self, session: StoredSession) -> str:
        turns = session.turns[-self.short_memory_turns :]
        if not turns:
            return ""

        history: list[str] = []
        for turn in turns:
            request_messages = [
                ChatMessage(**message)
                for message in turn.get("request", [])
                if isinstance(message, dict)
            ]
            user_text = _latest_user_prompt(request_messages)
            assistant_text = str(turn.get("response", "")).strip()
            if user_text:
                history.append(f"User: {user_text}")
            if assistant_text:
                history.append(f"Assistant: {assistant_text}")

        if not history:
            return ""
        return "\n".join(["Short-term conversation memory from Databricks-backed storage:", *history])

    @mlflow.trace
    async def _query_gateway(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, int | None]]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = await self.client.chat.completions.create(**kwargs)
        output = response.choices[0].message.content or ""
        usage = response.usage.model_dump() if response.usage else {}
        return output, usage


def _latest_user_prompt(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        content = _content_to_text(message.content).strip()
        if message.role == "user" and content:
            return content
    return ""


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"input_text", "text", "output_text"}:
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _gateway_role(role: str) -> str:
    if role in {"system", "user", "assistant", "tool"}:
        return role
    return "user"


def _base_url() -> str:
    value = os.getenv("LITELLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not value:
        raise RuntimeError("Set LITELLM_BASE_URL to your LiteLLM Gateway OpenAI-compatible base URL.")
    return value.rstrip("/")


def _optional_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    return int(value) if value else None
