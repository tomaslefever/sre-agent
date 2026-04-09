# Agent Documentation

## Use Cases
The AgentX Support Assistant is designed to handle L1 (Level 1) support queries autonomously. It has two primary use cases:
1. **Knowledge Retrieval:** Answering user queries by searching through a secure, private vector database containing company documentation.
2. **Issue Escalation:** Automatically generating support tickets when the user's issue falls outside the available documentation or when the user explicitly requests human assistance.

## Implementation Details
The agent is implemented using LangChain's `create_tool_calling_agent`. We utilize OpenAI's `gpt-4o-mini` as the reasoning engine due to its speed, cost-effectiveness, and excellent tool-calling capabilities.

### Tools Provided
* `buscar_en_base_de_conocimiento`: Connects to a Qdrant Vector Store to perform semantic search using `text-embedding-3-small` embeddings.
* `crear_ticket_soporte`: A mocked integration tool that simulates the creation of a Jira/Zendesk ticket, returning a structured ticket ID. *Note: Mocked as per hackathon rules.*

## Observability Evidence
We integrated **Arize Phoenix** via the OpenInference standard. Every interaction in the application generates a trace that captures:
* The exact prompt sent to the LLM.
* Which tool the agent decided to use and the parameters it extracted from the user's input.
* Retrieval latency from Qdrant.
* Total token consumption per request.
*(During evaluation, reviewers can inspect these traces live at `http://localhost:6006`)*

## Safety Measures
* **Deterministic Configuration:** The LLM temperature is set to `0` to reduce hallucinations and ensure predictable tool usage.
* **System Prompt Guardrails:** The agent is explicitly instructed via the system prompt to *only* rely on the provided tools to answer domain-specific questions, preventing it from inventing company policies.