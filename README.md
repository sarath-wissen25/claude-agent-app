# LiteLLM Gateway Agent on Databricks Apps + MLflow Agent Server

This project is ready for the Databricks Apps hosting path. Databricks Apps runs the app process, and MLflow Agent Server supplies the compatible agent server surface for invocations, the chat proxy, validation, and MLflow tracing. The agent calls your LiteLLM Gateway through its OpenAI-compatible `/v1` API.

## Files

- `app/main.py` - Databricks Apps entrypoint that starts MLflow Agent Server.
- `app/gateway_agent.py` - LiteLLM Gateway orchestration and MLflow-traced agent logic.
- `app/memory.py` - short-term conversation memory with Databricks SQL/Delta support and local fallback.
- `app/schemas.py` - request/response models used by the agent layer.
- `app.yaml` - Databricks Apps process configuration.
- `agent.py` - compatibility import for the Databricks Apps gateway agent.

## Hosting Model

Use Databricks Apps for this repo. The app command is:

```bash
python -m app.main
```

The app starts MLflow Agent Server, so you do not need to write a full FastAPI backend by hand. Agent requests should use the Responses-compatible payload shape:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "My project code name is Lakehouse Indigo. Remember it."
    }
  ],
  "custom_inputs": {
    "conversation_id": "demo-session-1",
    "user_id": "demo-user"
  }
}
```

Use the same `custom_inputs.conversation_id` for follow-up calls to reuse short-term memory.

## Required LiteLLM Gateway Environment Variables

```bash
LITELLM_BASE_URL="https://your-litellm-gateway.example.com/v1"
LITELLM_API_KEY="..."
LITELLM_MODEL="gpt-4o-mini"
```

Optional model settings:

```bash
LITELLM_TEMPERATURE="0.2"
LITELLM_MAX_TOKENS="4096"
LITELLM_TIMEOUT_SECONDS="120"
AGENT_SYSTEM_PROMPT="You are a Databricks-hosted assistant."
```

`LITELLM_BASE_URL` should include the OpenAI-compatible `/v1` suffix exposed by your LiteLLM Gateway.

## Short-Term Memory

The default memory mode is `auto`:

```bash
SHORT_MEMORY_BACKEND="auto"
SHORT_MEMORY_MAX_TURNS="12"
```

In Databricks, set these to enable durable Delta-backed short-term memory:

```bash
SHORT_MEMORY_BACKEND="databricks-sql"
SHORT_MEMORY_TABLE="main.default.litellm_agent_short_memory"
DATABRICKS_WAREHOUSE_ID="<serverless-or-pro-sql-warehouse-id>"
```

The table is created automatically if it does not exist. The app stores recent turns by `conversation_id` and injects the last `SHORT_MEMORY_MAX_TURNS` into the gateway request as short-term memory.

For local development, or if Databricks SQL settings are not present, memory falls back to:

```bash
SHORT_MEMORY_LOCAL_PATH="/tmp/litellm-agent-sessions.json"
```

Local JSON memory is only for development; use the Databricks SQL/Delta backend for hosted apps.

## Deploy To Databricks Apps

1. Create or select a Databricks App.
2. Sync this repository folder to the app source.
3. Configure app environment variables/secrets for LiteLLM Gateway access and optional memory settings.
4. Start the app with `app.yaml`.
5. Invoke the MLflow Agent Server endpoint exposed by the Databricks App.

## Observability

`app/gateway_agent.py` decorates the agent run and LiteLLM Gateway call with `mlflow.trace`, so Databricks captures traces for agent invocations. The response `custom_outputs` includes:

- `conversation_id`
- `session_id`
- `turn_count`
- `latency_ms`
- `model`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `memory_backend`

## Local Smoke Test

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Agent Server locally:

```bash
python -m app.main
```

Then send a Responses-compatible request to the local `/invocations` endpoint exposed by MLflow Agent Server.
