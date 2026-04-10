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

vector_store = get_vector_store()

@tool
def buscar_conocimiento(query: str) -> str:
    """Importante: Usa esto para buscar cualquier error en los archivos '.log' subidos, síntomas o manuales de infraestructura."""
    docs = vector_store.as_retriever(search_kwargs={"k": 5}).invoke(query)
    resultados = []
    for d in docs:
        origen = d.metadata.get("source", "kb_general")
        resultados.append(f"[{origen}]: {d.page_content}")
    return "\n\n---\n\n".join(resultados) if resultados else "No se encontró coincidencia relevante."

@tool
def listar_archivos_conocimiento(repo_filtro: str = None) -> str:
    """Devuelve la lista de archivos únicos indexados en la base de conocimiento. Úsala para saber qué archivos puedes leer."""
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
    client = QdrantClient(url=qdrant_url)
    pages = client.scroll(collection_name="kb_sre", limit=100, with_payload=True)[0]
    archivos = set()
    for p in pages:
        src = p.payload.get("source", "desconocido")
        if not repo_filtro or (repo_filtro in src):
            archivos.add(src)
    if not archivos:
        return "No hay archivos indexados aún en la base de conocimiento."
    return "Archivos disponibles:\n- " + "\n- ".join(sorted(list(archivos)))

@tool
def leer_archivo_conocimiento(nombre_archivo: str) -> str:
    """Recupera el contenido completo (o pedazos principales) de un archivo específico por su nombre exacto."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
    client = QdrantClient(url=qdrant_url)
    res = client.scroll(
        collection_name="kb_sre",
        scroll_filter=Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=nombre_archivo))]),
        limit=20,
        with_payload=True
    )[0]
    if not res:
        return f"No se encontró el archivo '{nombre_archivo}'."
    contenido = "\n[Continuación...]\n".join([p.payload.get("page_content", "") for p in res])
    return f"Contenido de {nombre_archivo}:\n\n{contenido}"

@tool
def crear_ticket_sre(reporte: str, autor: str, asignado: str = None) -> str:
    """Crea un ticket en el sistema."""
    if not asignado:
        import random
        asignado = random.choice(TECNICOS)
    t_id = f"TCK-{uuid.uuid4().hex[:6].upper()}"
    db = SessionLocal()
    try:
        new_ticket = Ticket(id=t_id, report=reporte, author=autor, assigned_to=asignado)
        db.add(new_ticket)
        db.flush()
        if "last_upload" in st.session_state and st.session_state.last_upload:
            att = Attachment(id=str(uuid.uuid4()), ticket_id=t_id, 
                             filename=st.session_state.last_upload["name"],
                             file_type=st.session_state.last_upload["type"])
            db.add(att)
            st.session_state.last_upload = None
        db.commit()
        res = f"✅ Ticket {t_id} creado y asignado a {asignado}."
    except Exception as e:
        db.rollback()
        res = f"❌ Error: {str(e)}"
    finally:
        db.close()
    return res

@tool
def leer_ticket(ticket_id: str) -> str:
    """Lee toda la información actual de un ticket."""
    db = SessionLocal()
    t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if t:
        info = f"ID: {t.id}\nStatus: {t.status}\nAutor: {t.author}\nAsignado: {t.assigned_to}\nReporte: {t.report}\nVeredicto: {t.veredicto or 'N/A'}"
        db.close()
        return info
    db.close()
    return "Ticket no encontrado."

@tool
def actualizar_veredicto(ticket_id: str, veredicto_txt: str) -> str:
    """Actualiza o define el análisis y veredicto actual del ticket."""
    db = SessionLocal()
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if ticket:
        ticket.veredicto = veredicto_txt
        db.commit()
        db.close()
        return f"✅ Veredicto actualizado en {ticket_id}."
    db.close()
    return "Ticket no encontrado."

@tool
def generar_plan_accion(ticket_id: str, nuevo_plan: str) -> str:
    """Agrega una nueva versión del plan de acción."""
    db = SessionLocal()
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if ticket:
        planes = list(ticket.planes_accion) if ticket.planes_accion else []
        planes.append({"version": len(planes)+1, "plan": nuevo_plan, "fecha": datetime.utcnow().isoformat()})
        ticket.planes_accion = planes
        db.commit()
        db.close()
        return f"✅ Plan generado para {ticket_id}."
    db.close()
    return "Ticket no encontrado."

@tool
def diagnostico_fast_track(ticket_id: str) -> str:
    """Análisis rápido con 10 chunks de contexto."""
    db = SessionLocal()
    t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not t:
        db.close()
        return "Ticket no encontrado."
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    v_store = get_vector_store()
    contextos = v_store.as_retriever(search_kwargs={"k": 10}).invoke(t.report)
    context_text = "\n---\n".join([d.page_content for d in contextos])
    fast_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    msg_prompt = f"Análisis rápido para {t.report}\nContexto:\n{context_text}\nResponde JSON con veredicto y plan."
    res = fast_llm.invoke(msg_prompt)
    try:
        import json
        data = json.loads(res.content.replace("```json", "").replace("```", "").strip())
        t.veredicto = data["veredicto"]
        planes = list(t.planes_accion) if t.planes_accion else []
        planes.append({"version": len(planes)+1, "plan": data["plan"], "fecha": datetime.utcnow().isoformat()})
        t.planes_accion = planes
        db.commit()
        db.close()
        return f"✅ Fast-Track completado para {ticket_id}."
    except Exception as e:
        db.close()
        return f"Error: {str(e)}"

@tool
def ejecutar_plan_accion(ticket_id: str) -> str:
    """Mueve a PENDING_NOTIF y añade hilo."""
    db = SessionLocal()
    t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if t:
        t.status = "PENDING_NOTIF"
        db.add(TicketThread(id=str(uuid.uuid4()), ticket_id=ticket_id, author="SRE-Agent", content="Plan ejecutado. Revisión pendiente."))
        db.commit()
        db.close()
        return f"🚀 Plan ejecutado para {ticket_id}."
    db.close()
    return "Ticket no encontrado."

tools = [buscar_conocimiento, listar_archivos_conocimiento, leer_archivo_conocimiento, leer_ticket, crear_ticket_sre, actualizar_veredicto, generar_plan_accion, diagnostico_fast_track, ejecutar_plan_accion]

def get_agent_executor():
    modelo_llm = os.getenv("OPENAI_MODEL", "gpt-4o")
    llm = ChatOpenAI(model=modelo_llm, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres AgentX, SRE automatizado."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    return AgentExecutor(agent=create_tool_calling_agent(llm, tools, prompt), tools=tools, verbose=True)
