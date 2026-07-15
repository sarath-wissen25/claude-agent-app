from app.main import DatabricksAppsGatewayAgent


if __name__ == "__main__":
    agent = DatabricksAppsGatewayAgent()
    response = agent.predict(
        {
            "messages": [
                {"role": "user", "content": "Hello. Confirm you are running as a Databricks Apps LiteLLM Gateway agent."}
            ],
            "custom_inputs": {"conversation_id": "local-demo-session"},
        }
    )
    print(response)
