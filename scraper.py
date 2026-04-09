import os
import requests
import uuid
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from functools import lru_cache

# Using the same imports as app to connect to DB/Qdrant
from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

DB_URL = os.getenv("DATABASE_URL", "postgresql://agentx:supersecret@postgres:5432/chat_history")
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

def get_qdrant():
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
    client = QdrantClient(url=qdrant_url)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return QdrantVectorStore(client=client, collection_name="kb_sre", embedding=embeddings)

def sync_github_repo(repo_url: str):
    """Fetches text files from a Github repo and pushes them to Qdrant."""
    token = os.getenv("GITHUB_TOKEN", "")
    headers = {"Authorization": f"token {token}"} if token else {}
    
    # Parse URL
    clean_url = repo_url.replace("https://github.com/", "").replace(".git", "").strip()
    if clean_url.endswith("/"):
        clean_url = clean_url[:-1]
    
    # Fetch default branch
    api_url = f"https://api.github.com/repos/{clean_url}"
    repo_res = requests.get(api_url, headers=headers)
    if repo_res.status_code != 200:
        return {"status": "error", "message": f"Cannot find repo {repo_url}"}
    
    default_branch = repo_res.json().get("default_branch", "main")
    
    # Fetch tree
    tree_url = f"https://api.github.com/repos/{clean_url}/git/trees/{default_branch}?recursive=1"
    tree_res = requests.get(tree_url, headers=headers)
    if tree_res.status_code != 200:
        return {"status": "error", "message": f"Cannot fetch tree for {repo_url}"}
    
    tree_data = tree_res.json().get("tree", [])
    valid_exts = ('.md', '.py', '.txt', '.js', '.ts', '.json', '.yml', '.yaml', '.sh')
    
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    
    for item in tree_data:
        if item["type"] == "blob" and item["path"].endswith(valid_exts):
            raw_url = f"https://raw.githubusercontent.com/{clean_url}/{default_branch}/{item['path']}"
            file_res = requests.get(raw_url, headers=headers)
            if file_res.status_code == 200:
                content = file_res.text
                chunks = text_splitter.split_text(content)
                for chunk in chunks:
                    docs.append(Document(page_content=chunk, metadata={"source": f"{repo_url}/{item['path']}", "repo": repo_url}))
                    
    if docs:
        vector_store = get_qdrant()
        # Idealmente limpiaríamos el repo viejo en Qdrant, pero para hackathon simplemente adosamos.
        vector_store.add_documents(docs)
        
    return {"status": "ok", "docs_indexed": len(docs)}

