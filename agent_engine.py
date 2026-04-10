import os
import uuid
import streamlit as st
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from database import SessionLocal, Ticket, TicketThread, Attachment

TECHNICIANS = ["Alex SRE", "Sonia DevOps", "Carlos Cloud", "Marta Security"]

@st.cache_resource
def get_vector_store():
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
    client = QdrantClient(url=qdrant_url)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    try:
        if not client.collection_exists("kb_sre"):
            client.create_collection(
                collection_name="kb_sre",
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
    except Exception:
        pass
    return QdrantVectorStore(client=client, collection_name="kb_sre", embedding=embeddings)


def diagnose_qdrant() -> dict:
    """Inspects Qdrant and returns a report on the vector database state."""
    q_url = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
    client = QdrantClient(url=q_url)

    result = {
        "collection_exists": False,
        "total_points": 0,
        "files": {},
        "metadata_keys": set(),
        "sample_payloads": []
    }

    try:
        if not client.collection_exists("kb_sre"):
            return result

        result["collection_exists"] = True
        info = client.get_collection("kb_sre")
        result["total_points"] = info.points_count

        all_points = []
        offset = None
        while True:
            batch, next_offset = client.scroll(
                collection_name="kb_sre", limit=100, with_payload=True, offset=offset
            )
            all_points.extend(batch)
            if next_offset is None or len(all_points) >= 500:
                break
            offset = next_offset

        for p in all_points:
            payload = p.payload or {}
            result["metadata_keys"].update(payload.keys())
            if "metadata" in payload and isinstance(payload["metadata"], dict):
                result["metadata_keys"].update(f"metadata.{k}" for k in payload["metadata"].keys())

            src = (
                payload.get("source")
                or payload.get("metadata", {}).get("source")
                or "NO_SOURCE"
            )
            result["files"][src] = result["files"].get(src, 0) + 1

        for p in all_points[:3]:
            sample = {}
            payload = p.payload or {}
            for k, v in payload.items():
                if k == "page_content" or k == "content":
                    sample[k] = str(v)[:200] + "..."
                else:
                    sample[k] = v
            result["sample_payloads"].append(sample)

        result["metadata_keys"] = list(result["metadata_keys"])

    except Exception as e:
        result["error"] = str(e)

    return result


@tool
def search_knowledge(query: str) -> str:
    """Searches the vector knowledge base. Returns fragments with source file and lines."""
    v_store = get_vector_store()
    docs = v_store.as_retriever(search_kwargs={"k": 8}).invoke(query)
    results = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        start = d.metadata.get("start_index", "?")
        chunk_id = d.metadata.get("chunk_id", "?")
        results.append(f"[FILE: {src} | chunk:{chunk_id} | offset:{start}]\n{d.page_content}")
    return "\n\n---\n\n".join(results) if results else "Nothing found."

@tool
def list_knowledge_files(repo_filter: str = None) -> str:
    """Lists ALL unique files indexed in Qdrant. Use this to see what code you have available before reading."""
    q_url = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
    client = QdrantClient(url=q_url)
    all_points = []
    offset = None
    while True:
        batch, next_offset = client.scroll(collection_name="kb_sre", limit=100, with_payload=True, offset=offset)
        all_points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset
    files = {}
    for p in all_points:
        src = p.payload.get("source", p.payload.get("metadata", {}).get("source", "unknown"))
        if repo_filter and repo_filter not in src:
            continue
        if src not in files:
            files[src] = 0
        files[src] += 1
    lines = [f"- {name} ({count} chunks)" for name, count in sorted(files.items())]
    return f"Total: {len(files)} indexed files\n" + "\n".join(lines)

@tool
def read_knowledge_file(file_name: str) -> str:
    """Reads ALL chunks of a specific file. Returns the full reconstructed content with position markers."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    q_url = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
    client = QdrantClient(url=q_url)

    for filter_key in ["metadata.source", "source"]:
        res = client.scroll(
            collection_name="kb_sre",
            scroll_filter=Filter(must=[FieldCondition(key=filter_key, match=MatchValue(value=file_name))]),
            limit=50,
            with_payload=True
        )[0]
        if res:
            break

    if not res:
        return f"File '{file_name}' not found. Use 'list_knowledge_files' to see available files."

    chunks = []
    for i, p in enumerate(res):
        content = p.payload.get("page_content", p.payload.get("content", ""))
        offset = p.payload.get("start_index", p.payload.get("metadata", {}).get("start_index", "?"))
        chunks.append(f"--- [Chunk {i+1}/{len(res)} | offset:{offset}] ---\n{content}")

    return f"FILE: {file_name} ({len(res)} fragments)\n\n" + "\n\n".join(chunks)

@tool
def detailed_code_search(query: str, limit: int = 20) -> str:
    """Deep code search: Gets up to 20 relevant fragments with full metadata (file, position). Use when you need more context than search_knowledge."""
    v_store = get_vector_store()
    k = min(limit, 30)
    docs = v_store.as_retriever(search_kwargs={"k": k}).invoke(query)

    by_file = {}
    for d in docs:
        src = d.metadata.get("source", d.metadata.get("metadata", {}).get("source", "unknown"))
        offset = d.metadata.get("start_index", "?")
        if src not in by_file:
            by_file[src] = []
        by_file[src].append({"offset": offset, "content": d.page_content})

    report = []
    for file_path, fragments in by_file.items():
        report.append(f"\n### FILE: {file_path} ({len(fragments)} matches)")
        for f in fragments:
            report.append(f"  [offset:{f['offset']}]\n{f['content']}")

    return f"Deep search: {len(docs)} results in {len(by_file)} files\n" + "\n---\n".join(report)

@tool
def create_sre_ticket(report: str, author: str, assigned_to: str = None) -> str:
    """Creates a ticket."""
    if not assigned_to:
        import random
        assigned_to = random.choice(TECHNICIANS)
    t_id = f"TCK-{uuid.uuid4().hex[:6].upper()}"
    current_session = st.session_state.get("session_id") if hasattr(st, "session_state") else None
    db = SessionLocal()
    try:
        db.add(Ticket(id=t_id, report=report, author=author, assigned_to=assigned_to, session_id=current_session))
        db.commit()
        return f"Ticket {t_id} created."
    except Exception as e:
        db.rollback()
        return f"Error: {str(e)}"
    finally:
        db.close()

@tool
def read_ticket(ticket_id: str) -> str:
    """Reads ticket info."""
    db = SessionLocal()
    t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    db.close()
    return f"ID: {t.id}, Status: {t.status}" if t else "Not found."

@tool
def update_verdict(ticket_id: str, verdict_text: str) -> str:
    """Updates verdict."""
    db = SessionLocal()
    t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if t:
        t.verdict = verdict_text
        db.commit()
    db.close()
    return "OK"

@tool
def generate_action_plan(ticket_id: str, new_plan: str) -> str:
    """Generates action plan."""
    db = SessionLocal()
    t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if t:
        plans = list(t.action_plans) if t.action_plans else []
        plans.append({"version": len(plans)+1, "plan": new_plan, "date": datetime.utcnow().isoformat()})
        t.action_plans = plans
        db.commit()
    db.close()
    return "OK"

@tool
def fast_track_diagnosis(ticket_id: str) -> str:
    """Fast track: Searches 10 chunks in Qdrant, generates diagnosis + plan in a single pass with GPT-4o."""
    import json
    from langchain_core.messages import SystemMessage, HumanMessage

    db = SessionLocal()
    t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not t:
        db.close()
        return "Ticket not found."

    v_store = get_vector_store()
    ctx = v_store.as_retriever(search_kwargs={"k": 10}).invoke(t.report)

    context_parts = []
    for d in ctx:
        src = d.metadata.get("source", d.metadata.get("metadata", {}).get("source", "unknown"))
        offset = d.metadata.get("start_index", "?")
        context_parts.append(f"[FILE: {src} | offset:{offset}]\n{d.page_content}")
    context_text = "\n---\n".join(context_parts) if context_parts else "No context available in the knowledge base."

    fast_llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)
    messages = [
        SystemMessage(content="You are a Senior SRE Engineer expert in rapid incident diagnosis. Always respond in valid JSON."),
        HumanMessage(content=f"""Analyze this incident using ONLY the attached code/documentation context.

IMPORTANT: Base your analysis ONLY on the source code provided below. Do NOT invent files, logs or information not in the context. If a fragment is not relevant to the incident, ignore it.

REPORTED INCIDENT (TICKET {ticket_id}):
{t.report}

KNOWLEDGE BASE CONTEXT (code and documentation):
{context_text}

Respond ONLY with valid JSON with these keys:
{{
  "verdict": "Detailed technical explanation of the root cause, referencing ONLY files and code sections from the provided context",
  "files_reviewed": ["list of files from context you analyzed - use exact names from the FILE field"],
  "findings": ["list of specific findings with reference to the file and code fragment where you found each issue"],
  "plan": "Numbered concrete steps to resolve the incident, indicating which files to modify and what changes to make"
}}""")
    ]

    try:
        res = fast_llm.invoke(messages)
        raw = res.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        verdict = data.get("verdict", "Could not determine root cause.")
        plan = data.get("plan", "Could not generate a plan.")
        files_reviewed = data.get("files_reviewed", [])
        findings = data.get("findings", [])

        report_text = f"**Verdict:** {verdict}\n\n"
        if files_reviewed:
            report_text += f"**Files reviewed ({len(files_reviewed)}):**\n"
            for a in files_reviewed:
                report_text += f"- `{a}`\n"
        if findings:
            report_text += f"\n**Findings ({len(findings)}):**\n"
            for i, h in enumerate(findings, 1):
                report_text += f"{i}. {h}\n"
        report_text += f"\n**Action Plan:** {plan}"

        t.verdict = verdict
        plans = list(t.action_plans) if t.action_plans else []
        new_v = len(plans) + 1
        plans.append({
            "version": new_v,
            "plan": plan,
            "files_reviewed": files_reviewed,
            "findings": findings,
            "date": datetime.utcnow().isoformat()
        })
        t.action_plans = plans
        t.status = "IN_PROGRESS"

        db.add(TicketThread(
            id=str(uuid.uuid4()),
            ticket_id=ticket_id,
            author="SRE-Agent",
            content=f"**Fast-Track completed**\n\n{report_text}"
        ))
        db.commit()
        db.close()

        return report_text

    except json.JSONDecodeError as e:
        db.close()
        return f"Error: LLM did not return valid JSON. Raw response: {res.content[:500]}"
    except Exception as e:
        db.close()
        return f"Error in Fast-Track: {str(e)}"


@tool
def execute_action_plan(ticket_id: str) -> str:
    """Executes the ticket's action plan: generates corrected code with AI and pushes it to a branch on GitHub. Does NOT create a PR."""
    import json
    import requests
    from langchain_core.messages import SystemMessage, HumanMessage

    gh_token = os.getenv("GITHUB_TOKEN")
    if not gh_token:
        return "Error: GITHUB_TOKEN not configured."

    db = SessionLocal()
    t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not t:
        db.close()
        return "Ticket not found."

    plans = list(t.action_plans) if t.action_plans else []
    if not plans:
        db.close()
        return "No action plan. Generate an Action Plan first."

    latest_plan = plans[-1]
    plan_text = latest_plan.get("plan", "")
    files_reviewed = latest_plan.get("files_reviewed", [])
    findings = latest_plan.get("findings", [])

    from database import Repository
    repo = db.query(Repository).first()
    if not repo or not repo.url:
        db.close()
        return "No linked repository. Add one in Knowledge Base."

    repo_url = repo.url.rstrip("/")
    parts = repo_url.replace("https://github.com/", "").replace("http://github.com/", "").split("/")
    if len(parts) < 2:
        db.close()
        return f"Invalid repo URL: {repo_url}"
    owner, repo_name = parts[0], parts[1].replace(".git", "")

    headers = {
        "Authorization": f"token {gh_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    api_base = f"https://api.github.com/repos/{owner}/{repo_name}"

    import re
    verdict_slug = ""
    if t.verdict:
        slug = re.sub(r'[^a-z0-9\s]', '', t.verdict[:60].lower())
        verdict_slug = "-".join(slug.split()[:5])
    elif plan_text:
        slug = re.sub(r'[^a-z0-9\s]', '', plan_text[:60].lower())
        verdict_slug = "-".join(slug.split()[:5])
    branch_name = f"fix/{ticket_id.lower()}/{verdict_slug}" if verdict_slug else f"fix/{ticket_id.lower()}"

    try:
        ref_res = requests.get(f"{api_base}/git/ref/heads/main", headers=headers)
        if ref_res.status_code != 200:
            ref_res = requests.get(f"{api_base}/git/ref/heads/master", headers=headers)
        if ref_res.status_code != 200:
            db.close()
            return f"Error getting main branch: {ref_res.json().get('message', ref_res.status_code)}"

        base_sha = ref_res.json()["object"]["sha"]

        create_ref = requests.post(f"{api_base}/git/refs", headers=headers, json={
            "ref": f"refs/heads/{branch_name}",
            "sha": base_sha
        })
        if create_ref.status_code not in (200, 201, 422):
            db.close()
            return f"Error creating branch: {create_ref.json().get('message', '')}"

        v_store = get_vector_store()
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)

        modified_files = []

        for file_path in files_reviewed:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            q_url = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
            client = QdrantClient(url=q_url)

            code_chunks = []
            for filter_key in ["metadata.source", "source"]:
                res_q = client.scroll(
                    collection_name="kb_sre",
                    scroll_filter=Filter(must=[FieldCondition(key=filter_key, match=MatchValue(value=file_path))]),
                    limit=50, with_payload=True
                )[0]
                if res_q:
                    code_chunks = [p.payload.get("page_content", p.payload.get("content", "")) for p in res_q]
                    break

            if not code_chunks:
                continue

            original_code = "\n".join(code_chunks)

            fix_messages = [
                SystemMessage(content="You are a senior software engineer. Your task is to apply fixes to source code. Return ONLY the complete corrected code, no markdown, no explanations."),
                HumanMessage(content=f"""FILE: {file_path}

ORIGINAL CODE:
{original_code}

REPORTED ISSUE:
{t.report}

FINDINGS:
{chr(10).join(findings) if findings else 'See plan'}

FIX PLAN:
{plan_text}

Return the complete corrected code for the file. Only the code, no backticks or explanations.""")
            ]

            fix_res = llm.invoke(fix_messages)
            corrected_code = fix_res.content.strip()
            if corrected_code.startswith("```"):
                lines = corrected_code.split("\n")
                corrected_code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            clean_path = file_path.lstrip("/")
            file_res = requests.get(f"{api_base}/contents/{clean_path}?ref={branch_name}", headers=headers)

            verdict_short = (t.verdict[:80] + "...") if t.verdict and len(t.verdict) > 80 else (t.verdict or "automatic fix")
            commit_data = {
                "message": f"fix({ticket_id}): {clean_path}\n\n{verdict_short}\n\nRef: {ticket_id}",
                "content": __import__("base64").b64encode(corrected_code.encode()).decode(),
                "branch": branch_name
            }
            if file_res.status_code == 200:
                commit_data["sha"] = file_res.json()["sha"]

            put_res = requests.put(f"{api_base}/contents/{clean_path}", headers=headers, json=commit_data)
            if put_res.status_code in (200, 201):
                modified_files.append(clean_path)

        t.status = "PENDING_NOTIF"
        db.add(TicketThread(
            id=str(uuid.uuid4()), ticket_id=ticket_id, author="SRE-Agent",
            content=f"**Code generated on branch `{branch_name}`**\n\nModified files: {', '.join(modified_files)}"
        ))
        db.commit()
        db.close()
        return f"Code pushed to branch {branch_name}\nFiles: {', '.join(modified_files)}\n\nUse 'Send PR' to create the Pull Request."

    except Exception as e:
        t.status = "PENDING_NOTIF"
        db.add(TicketThread(
            id=str(uuid.uuid4()), ticket_id=ticket_id, author="SRE-Agent",
            content=f"Error executing plan: {str(e)}"
        ))
        db.commit()
        db.close()
        return f"Error: {str(e)}"


@tool
def create_pr_ticket(ticket_id: str) -> str:
    """Creates a Pull Request on GitHub for the ticket's fix branch. Must be run after execute_action_plan."""
    import requests

    gh_token = os.getenv("GITHUB_TOKEN")
    if not gh_token:
        return "Error: GITHUB_TOKEN not configured."

    db = SessionLocal()
    t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not t:
        db.close()
        return "Ticket not found."

    from database import Repository
    repo = db.query(Repository).first()
    if not repo or not repo.url:
        db.close()
        return "No linked repository."

    repo_url = repo.url.rstrip("/")
    parts = repo_url.replace("https://github.com/", "").replace("http://github.com/", "").split("/")
    if len(parts) < 2:
        db.close()
        return f"Invalid repo URL: {repo_url}"
    owner, repo_name = parts[0], parts[1].replace(".git", "")

    headers = {
        "Authorization": f"token {gh_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    api_base = f"https://api.github.com/repos/{owner}/{repo_name}"
    branch_prefix = f"fix/{ticket_id.lower()}"

    refs_res = requests.get(f"{api_base}/git/matching-refs/heads/{branch_prefix}", headers=headers)
    if refs_res.status_code != 200 or not refs_res.json():
        db.close()
        return f"No branch found for {ticket_id}. Execute the plan first."
    branch_name = refs_res.json()[-1]["ref"].replace("refs/heads/", "")

    base_check = requests.get(f"{api_base}/git/ref/heads/main", headers=headers)
    base_branch = "main" if base_check.status_code == 200 else "master"

    plans = list(t.action_plans) if t.action_plans else []
    plan_text = plans[-1].get("plan", "N/A") if plans else "N/A"

    threads = db.query(TicketThread).filter(TicketThread.ticket_id == ticket_id).all()
    files_info = ""
    for h in threads:
        if "Modified files:" in h.content:
            files_info = h.content.split("Modified files:")[-1].strip()
            break

    pr_body = f"""## Ticket: {ticket_id}

### Report
{t.report[:500]}

### Verdict
{t.verdict or 'N/A'}

### Modified files
{files_info or 'See commits on the branch'}

### Applied plan
{plan_text}

---
*Automatically generated by AgentX SRE*"""

    try:
        existing_prs = requests.get(f"{api_base}/pulls?head={owner}:{branch_name}&state=open", headers=headers)
        if existing_prs.status_code == 200 and existing_prs.json():
            pr_url = existing_prs.json()[0].get("html_url", "")
            t.status = "AWAITING_VALIDATION"
            db.add(TicketThread(
                id=str(uuid.uuid4()), ticket_id=ticket_id, author="SRE-Agent",
                content=f"**PR already exists:** [{branch_name}]({pr_url})\n\nTicket awaiting validation."
            ))
            db.commit()
            db.close()
            return f"PR already exists: {pr_url}"

        import re
        title_desc = re.sub(r'[\r\n]+', ' ', t.verdict[:70]) if t.verdict else "Automatic fix"
        pr_title = f"fix({ticket_id}): {title_desc}"

        pr_res = requests.post(f"{api_base}/pulls", headers=headers, json={
            "title": pr_title,
            "body": pr_body,
            "head": branch_name,
            "base": base_branch
        })

        if pr_res.status_code in (200, 201):
            pr_url = pr_res.json().get("html_url", "")
            t.status = "AWAITING_VALIDATION"
            db.add(TicketThread(
                id=str(uuid.uuid4()), ticket_id=ticket_id, author="SRE-Agent",
                content=f"**PR created:** [{branch_name}]({pr_url})\n\nTicket awaiting validation."
            ))
            db.commit()
            db.close()
            return f"PR created successfully: {pr_url}"
        else:
            resp_json = pr_res.json()
            err = resp_json.get("message", str(pr_res.status_code))
            errors_detail = resp_json.get("errors", [])
            detail = "; ".join([e.get("message", str(e)) for e in errors_detail]) if errors_detail else ""
            full_err = f"{err}. {detail}" if detail else err
            db.add(TicketThread(
                id=str(uuid.uuid4()), ticket_id=ticket_id, author="SRE-Agent",
                content=f"**Error creating PR:** {full_err}\n\nBranch: `{branch_name}`"
            ))
            db.commit()
            db.close()
            return f"Error creating PR: {full_err}"

    except Exception as e:
        db.add(TicketThread(
            id=str(uuid.uuid4()), ticket_id=ticket_id, author="SRE-Agent",
            content=f"**Error creating PR:** {str(e)}"
        ))
        db.commit()
        db.close()
        return f"Error: {str(e)}"

tools = [search_knowledge, list_knowledge_files, read_knowledge_file, detailed_code_search, read_ticket, create_sre_ticket, update_verdict, generate_action_plan, fast_track_diagnosis, execute_action_plan, create_pr_ticket]

def get_agent_executor():
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are AgentX, an automated L1/L2 SRE Engineer.
Available technicians: Alex SRE, Sonia DevOps, Carlos Cloud, Marta Security.

Your mandatory workflow:
1. Extract severity and affected system from the report.
2. ALWAYS use 'list_knowledge_files' to see what code you have available.
3. Use 'search_knowledge' for a quick initial search.
4. ALWAYS go deeper with 'detailed_code_search' (limit=20 or more) for full context.
5. Use 'read_knowledge_file' to read complete suspicious files. Read ALL relevant files regardless of how many there are.
6. Create tickets with detailed verdicts and action plans.

SEARCH RULES (CRITICAL):
- NEVER settle for a single search. Run multiple searches with different terms.
- If you find relevant files, READ THEM COMPLETELY with 'read_knowledge_file'.
- Search exhaustively regardless of how long it takes. Precision is more important than speed.
- If a result mentions other files, search and read those files too.

REPORT RULES:
- ALWAYS reference the files you reviewed by name.
- ALWAYS indicate the sections/lines/offsets where you found problems.
- Generate a structured report: Files Reviewed -> Findings -> Verdict -> Plan.

DATA ISOLATION:
- The vector database contains ONLY code from synced GitHub repositories. Use it freely.
- You do NOT have access to data from other tickets or sessions. Only work with the current incident's information.
- Attachments (images, logs) are provided as part of the user's message.

If given an image/screenshot description, analyze the visual evidence.
If given a ticket ID, use 'read_ticket' first."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    return AgentExecutor(agent=create_tool_calling_agent(llm, tools, prompt), tools=tools, verbose=True)


def get_ticket_agent(ticket_id: str, ticket_report: str, attachments_text: str = ""):
    """Creates a conversational agent contextualized for a specific ticket."""
    ticket_tools = [search_knowledge, list_knowledge_files, read_knowledge_file, detailed_code_search, read_ticket]
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are AgentX, an expert SRE assistant assigned to ticket {ticket_id}.

INCIDENT CONTEXT:
{ticket_report}

{f"INCIDENT ATTACHMENTS:{chr(10)}{attachments_text}" if attachments_text else ""}

You have access to the vector knowledge base to search and read source code.
Answer questions about the incident, analyze related code, and help investigate the root cause.
Search exhaustively in the vector database regardless of how long it takes. Read complete files when necessary."""),
        ("placeholder", "{{chat_history}}"),
        ("human", "{{input}}"),
        ("placeholder", "{{agent_scratchpad}}"),
    ])
    return AgentExecutor(agent=create_tool_calling_agent(llm, ticket_tools, prompt), tools=ticket_tools, verbose=True, max_iterations=25)


def analyze_image_with_vision(image_b64: str, mime_type: str, user_text: str = "") -> str:
    """Uses GPT-4o Vision to analyze an image and returns a technical description as text."""
    from langchain_core.messages import HumanMessage

    vision_llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0, max_tokens=1500)

    content_blocks = []
    if user_text:
        content_blocks.append({"type": "text", "text": user_text})
    content_blocks.append({"type": "text", "text": "Analyze this screenshot/image as an SRE Engineer. Describe exactly what you see: errors, logs, metrics, dashboards, stack traces, HTTP status codes, etc. Be extremely technical and precise."})
    content_blocks.append({
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}
    })

    msg = HumanMessage(content=content_blocks)
    response = vision_llm.invoke([msg])
    return response.content
