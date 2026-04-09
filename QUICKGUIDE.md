# Quick Start Guide

Follow these simple steps to run and test the application locally. The entire application runs inside Docker, ensuring a clean and reproducible environment.

### Prerequisites
* Docker and Docker Compose installed on your machine.
* An active OpenAI API Key.

### Step-by-Step Instructions

#### 1. Clone the repository
```bash
git clone [https://github.com/TU_USUARIO/mi-repo-agentx.git](https://github.com/TU_USUARIO/mi-repo-agentx.git)
cd mi-repo-agentx
```

#### 2. Set up environment variables
Copy the example environment file and add your actual API keys.
```bash
cp .env.example .env
```
*Open the `.env` file and replace the placeholder with your real OpenAI API Key.*

#### 3. Build and Run the Application
Start the entire stack (UI, Vector DB, Relational DB, and Observability) using Docker Compose:
```bash
docker compose up --build
```

#### 4. Access the Services
Once the containers are running, you can access the following interfaces in your browser:
* **Agent UI (Streamlit):** [http://localhost:8501](http://localhost:8501)
* **Observability Dashboard (Phoenix):** [http://localhost:6006](http://localhost:6006)
* **Vector DB (Qdrant - API):** [http://localhost:6333](http://localhost:6333)

**How to test the agent:** Ask a question about company knowledge (e.g., *"What is Qdrant?"*) or ask to create a support ticket (e.g., *"I need human help, please open a ticket"*). Try refreshing the page to see how PostgreSQL persists your chat history!