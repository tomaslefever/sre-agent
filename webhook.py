from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
import logging
from scraper import sync_github_repo, SessionLocal
from sqlalchemy import text

app = FastAPI(title="AgentX Webhook Listener")
logging.basicConfig(level=logging.INFO)

def run_sync_and_update_db(repo_url: str):
    logging.info(f"Starting background sync for {repo_url}")
    result = sync_github_repo(repo_url)
    logging.info(f"Sync result for {repo_url}: {result}")
    
    if result["status"] == "ok":
        try:
            db = SessionLocal()
            # Actualizamos la fecha
            from datetime import datetime
            query = text("UPDATE repositories SET last_updated = :now WHERE url = :url")
            db.execute(query, {"now": datetime.utcnow(), "url": repo_url})
            db.commit()
            db.close()
        except Exception as e:
            logging.error(f"Error updating DB: {e}")

@app.post("/webhook/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Webhook endpoint to be configured in GitHub.
    Listens for 'push' events and triggers repository re-sync.
    """
    payload = await request.json()
    
    # Extract repository URL from GitHub Webhook payload
    repo_data = payload.get("repository")
    if not repo_data:
        return {"status": "ignored", "reason": "No repository data in payload"}
        
    repo_url = repo_data.get("html_url")
    
    # Trigger background indexing so response is immediate to GitHub
    background_tasks.add_task(run_sync_and_update_db, repo_url)
    
    return {"status": "accepted", "message": f"Sync queued for {repo_url}", "docs_queued": True}

@app.get("/health")
def health():
    return {"status": "healthy"}
