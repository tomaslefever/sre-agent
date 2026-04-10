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

TECNICOS = ["Alex SRE", "Sonia DevOps", "Carlos Cloud", "Marta Security"]

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

@tool
def buscar_conocimiento(query: str) -> str:
    """Usa esto para buscar en la base de conocimiento."""
    v_store = get_vector_store()
    docs = v_store.as_retriever(search_kwargs={"k": 5}).invoke(query)
    resultados = []
    for d in docs:
        origen = d.metadata.get("source", "kb_general")
        resultados.append(f"[{origen}]: {d.page_content}")
    return "\n\n---\n\n".join(resultados) if resultados else "No se encontró nada."

@tool
def listar_archivos_conocimiento(repo_filtro: str = None) -> str:
    """Lista archivos disponibles."""
    q_url = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
    client = QdrantClient(url=q_url)
    res = client.scroll(collection_name="kb_sre", limit=100, with_payload=True)[0]
    archivos = {p.payload.get("source", "desconocido") for p in res}
    return "Archivos:\n- " + "\n- ".join(sorted(list(archivos)))

@tool
def leer_archivo_conocimiento(nombre_archivo: str) -> str:
    """Lee contenido de un archivo."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    q_url = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
    client = QdrantClient(url=q_url)
    res = client.scroll(
        collection_name="kb_sre",
        scroll_filter=Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=nombre_archivo))]),
        limit=20,
        with_payload=True
    )[0]
    return "\n".join([p.payload.get("page_content", "") for p in res])

@tool
def crear_ticket_sre(reporte: str, autor: str, asignado: str = None) -> str:
    """Crea un ticket."""
    if not asignado:
        import random
        asignado = random.choice(TECNICOS)
    t_id = f"TCK-{uuid.uuid4().hex[:6].upper()}"
    db = SessionLocal()
    try:
        db.add(Ticket(id=t_id, report=reporte, author=autor, assigned_to=asignado))
        db.commit()
        return f"Ticket {t_id} creado."
    except Exception as e:
        db.rollback()
        return f"Error: {str(e)}"
    finally:
        db.close()

@tool
def leer_ticket(ticket_id: str) -> str:
    """Lee info de ticket."""
    db = SessionLocal()
    t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    db.close()
    return f"ID: {t.id}, Estado: {t.status}" if t else "No encontrado."

@tool
def actualizar_veredicto(ticket_id: str, veredicto_txt: str) -> str:
    """Actualiza veredicto."""
    db = SessionLocal()
    t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if t:
        t.veredicto = veredicto_txt
        db.commit()
    db.close()
    return "OK"

@tool
def generar_plan_accion(ticket_id: str, nuevo_plan: str) -> str:
    """Genera plan."""
    db = SessionLocal()
    t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if t:
        planes = list(t.planes_accion) if t.planes_accion else []
        planes.append({"version": len(planes)+1, "plan": nuevo_plan, "fecha": datetime.utcnow().isoformat()})
        t.planes_accion = planes
        db.commit()
    db.close()
    return "OK"

@tool
def diagnostico_fast_track(ticket_id: str) -> str:
    """Diagnóstico rápido."""
    db = SessionLocal()
    t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not t:
        db.close()
        return "No encontrado."
    fast_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    v_store = get_vector_store()
    ctx = v_store.as_retriever(search_kwargs={"k": 10}).invoke(t.report)
    prompt = f"Analiza: {t.report}\nContexto: {' '.join([d.page_content for d in ctx])}\nResponde JSON."
    res = fast_llm.invoke(prompt)
    try:
        import json
        data = json.loads(res.content.replace("```json", "").replace("```", "").strip())
        t.veredicto = data["veredicto"]
        planes = list(t.planes_accion) if t.planes_accion else []
        planes.append({"version": len(planes)+1, "plan": data["plan"], "fecha": datetime.utcnow().isoformat()})
        t.planes_accion = planes
        db.commit()
    except:
        pass
    db.close()
    return "Fast-Track OK"

@tool
def ejecutar_plan_accion(ticket_id: str) -> str:
    """Ejecuta plan."""
    db = SessionLocal()
    t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if t:
        t.status = "PENDING_NOTIF"
        db.add(TicketThread(id=str(uuid.uuid4()), ticket_id=ticket_id, author="SRE-Agent", content="Plan ejecutado."))
        db.commit()
    db.close()
    return "Plan Ejecutado"

tools = [buscar_conocimiento, listar_archivos_conocimiento, leer_archivo_conocimiento, leer_ticket, crear_ticket_sre, actualizar_veredicto, generar_plan_accion, diagnostico_fast_track, ejecutar_plan_accion]

def get_agent_executor():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres AgentX."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    return AgentExecutor(agent=create_tool_calling_agent(llm, tools, prompt), tools=tools, verbose=True)
