# Quick Start Guide

Follow these simple steps to run and test the application locally. The entire application runs inside Docker, ensuring a clean and reproducible environment.

### Prerequisites
* Docker and Docker Compose installed on your machine.
* An active OpenAI API Key.

### Step-by-Step Instructions

#### 1. Clone the repository
```bash
git clone https://github.com/tomaslefever/sre-agent.git
cd sre-agent
```

#### 2. Set up environment variables
Copy the example environment file and add your actual API keys.
```bash
cp env.example .env
```
*Open the `.env` file and replace the placeholder with your real OpenAI API Key.*

#### 3. Build and Run (Local or Cloud)

##### Local Deployment
To run the entire infrastructure locally:
```bash
docker compose up --build
```

##### Cloud Deployment (Easypanel / Railway / etc.)
Does **NOT** deploy as individual Apps. Instead, use the **Stack** or **Project** feature:
1. Create a new **Stack**.
2. Paste the contents of `docker-compose.yml`.
3. Set the `OPENAI_API_KEY` in the environment variables of the Stack.
4. Deploy. This ensures all services (Qdrant, Postgres, Phoenix) are linked and healthy.

#### 4. Access the Services
Once everything is ready:
* **Agent UI:** [http://localhost:8501](http://localhost:8501)
* **Observability:** [http://localhost:6006](http://localhost:6006)
* **Vector DB:** [http://localhost:6333](http://localhost:6333)

**Important:** The `app` service now includes healthchecks. If you see it "starting", wait a few moments until the databases are "healthy" before trying to use the chat.