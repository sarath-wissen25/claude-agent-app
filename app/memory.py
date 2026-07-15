from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class StoredSession:
    conversation_id: str
    claude_session_id: str | None = None
    turns: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class SessionStore(Protocol):
    backend_name: str

    def get(self, conversation_id: str) -> StoredSession:
        ...

    def save(self, session: StoredSession) -> None:
        ...

    def list(self) -> list[StoredSession]:
        ...


class JsonSessionStore:
    backend_name = "local-json"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, conversation_id: str) -> StoredSession:
        data = self._read_all()
        raw = data.get(conversation_id)
        if raw is None:
            return StoredSession(conversation_id=conversation_id)
        return StoredSession(**raw)

    def save(self, session: StoredSession) -> None:
        data = self._read_all()
        session.updated_at = time.time()
        data[session.conversation_id] = asdict(session)
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)

    def list(self) -> list[StoredSession]:
        return [StoredSession(**raw) for raw in self._read_all().values()]

    def _read_all(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}


class DatabricksSqlSessionStore:
    backend_name = "databricks-sql"

    def __init__(self, table_name: str, warehouse_id: str | None = None) -> None:
        self.table_name = table_name
        self.warehouse_id = warehouse_id or os.getenv("DATABRICKS_WAREHOUSE_ID")
        if not self.warehouse_id:
            raise ValueError("DATABRICKS_WAREHOUSE_ID is required for Databricks SQL memory.")

        try:
            from databricks.sdk import WorkspaceClient
        except ImportError as exc:
            raise RuntimeError("databricks-sdk is required for Databricks SQL memory.") from exc

        self.client = WorkspaceClient()
        self._ensure_table()

    def get(self, conversation_id: str) -> StoredSession:
        rows = self._execute(
            f"SELECT payload FROM {self.table_name} WHERE conversation_id = :conversation_id LIMIT 1",
            {"conversation_id": conversation_id},
        )
        if not rows:
            return StoredSession(conversation_id=conversation_id)
        payload = rows[0][0]
        if isinstance(payload, str):
            return StoredSession(**json.loads(payload))
        return StoredSession(**payload)

    def save(self, session: StoredSession) -> None:
        session.updated_at = time.time()
        payload = json.dumps(asdict(session), sort_keys=True)
        self._execute(
            f"""
            MERGE INTO {self.table_name} target
            USING (
              SELECT :conversation_id AS conversation_id, :payload AS payload
            ) source
            ON target.conversation_id = source.conversation_id
            WHEN MATCHED THEN UPDATE SET payload = source.payload, updated_at = current_timestamp()
            WHEN NOT MATCHED THEN INSERT (conversation_id, payload, updated_at)
            VALUES (source.conversation_id, source.payload, current_timestamp())
            """,
            {"conversation_id": session.conversation_id, "payload": payload},
        )

    def list(self) -> list[StoredSession]:
        rows = self._execute(
            f"SELECT payload FROM {self.table_name} ORDER BY updated_at DESC LIMIT 100",
            {},
        )
        sessions = []
        for row in rows:
            payload = row[0]
            sessions.append(StoredSession(**json.loads(payload if isinstance(payload, str) else json.dumps(payload))))
        return sessions

    def _ensure_table(self) -> None:
        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
              conversation_id STRING NOT NULL,
              payload STRING NOT NULL,
              updated_at TIMESTAMP NOT NULL
            )
            USING DELTA
            """,
            {},
        )

    def _execute(self, statement: str, parameters: dict[str, Any]) -> list[list[Any]]:
        response = self.client.statement_execution.execute_statement(
            warehouse_id=self.warehouse_id,
            statement=statement,
            parameters=parameters,
            wait_timeout="30s",
        )
        if not response.result or not response.result.data_array:
            return []
        return response.result.data_array


def build_session_store() -> SessionStore:
    backend = os.getenv("SHORT_MEMORY_BACKEND", "auto").strip().lower()
    table_name = os.getenv("SHORT_MEMORY_TABLE", "").strip()
    local_path = os.getenv("SHORT_MEMORY_LOCAL_PATH", "/tmp/litellm-agent-sessions.json")

    if backend in {"databricks", "databricks-sql", "sql"}:
        return DatabricksSqlSessionStore(table_name=_required_table_name(table_name))
    if backend == "local-json":
        return JsonSessionStore(local_path)
    if backend == "auto" and table_name and os.getenv("DATABRICKS_WAREHOUSE_ID"):
        return DatabricksSqlSessionStore(table_name=table_name)
    return JsonSessionStore(local_path)


def _required_table_name(table_name: str) -> str:
    if not table_name:
        raise ValueError("SHORT_MEMORY_TABLE must be set, for example main.default.agent_short_memory.")
    return table_name
