# Scaling the Application

While the current architecture is containerized and production-ready for small workloads, scaling to serve thousands of concurrent users would require specific architectural upgrades.

## Current Assumptions
* The application runs on a single node (using Easypanel/Docker Compose).
* State (chat history) is managed in memory via Streamlit's session state.
* Qdrant and Phoenix run as local containers alongside the application.

## Scaling Decisions & Upgrades

### 1. Vector Database Scaling
* **Current:** Local Qdrant container with a file-mounted volume.
* **Production:** Migrate to **Qdrant Cloud** (managed service) or a distributed Qdrant cluster on Kubernetes. This removes the stateful burden from our application nodes and ensures high availability for semantic search.

### 2. Application & UI Scaling
* **Current:** Streamlit running in a single container.
* **Production:** Streamlit is stateful and not ideal for massive horizontal scaling. For a true enterprise deployment, we would decouple the frontend and backend:
  * **Backend:** Expose the LangChain agent via a **FastAPI** REST/WebSocket endpoint.
  * **Frontend:** Build a lightweight React/Next.js frontend.
  * **Deployment:** Deploy the FastAPI backend across multiple pods using Kubernetes (EKS/GKE) behind a Load Balancer.

### 3. State Management
* **Current:** Chat history is lost if the user refreshes the page.
* **Production:** Implement **Redis** or **PostgreSQL** to persist `ChatMessageHistory` per session ID, allowing users to resume conversations across different devices or sessions.

### 4. Tool Execution (De-mocking)
* **Current:** The `crear_ticket_soporte` tool is mocked.
* **Production:** Replace the mock with asynchronous HTTP clients (e.g., `httpx`) hitting the actual Zendesk/Jira APIs. Implement retry logic (Tenacity) to handle external API rate limits and timeouts gracefully.