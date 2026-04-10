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
    """Ruta rapida: Busca 10 chunks en Qdrant, genera diagnostico + plan en una sola pasada con GPT-4o."""
    import json
    from langchain_core.messages import SystemMessage, HumanMessage
    
    db = SessionLocal()
    t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not t:
        db.close()
        return "Ticket no encontrado."
    
    # 1. Buscar contexto
    v_store = get_vector_store()
    ctx = v_store.as_retriever(search_kwargs={"k": 10}).invoke(t.report)
    context_text = "\n---\n".join([d.page_content for d in ctx]) if ctx else "Sin contexto disponible en la base de conocimiento."
    
    # 2. Prompt estructurado
    fast_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    messages = [
        SystemMessage(content="Eres un Ingeniero SRE Senior experto en diagnostico rapido de incidentes. Siempre respondes en JSON valido."),
        HumanMessage(content=f"""Analiza este incidente usando el contexto del codigo/documentacion adjunto.

INCIDENTE REPORTADO:
{t.report}

CONTEXTO DE LA BASE DE CONOCIMIENTO:
{context_text}

Responde UNICAMENTE con un JSON valido con estas dos claves:
{{
  "veredicto": "Explicacion tecnica detallada de la causa raiz del problema",
  "plan": "Pasos numerados y concretos para resolver el incidente"
}}""")
    ]
    
    try:
        res = fast_llm.invoke(messages)
        raw = res.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        
        veredicto = data.get("veredicto", "No se pudo determinar la causa raiz.")
        plan = data.get("plan", "No se pudo generar un plan.")
        
        # Guardar en DB
        t.veredicto = veredicto
        planes = list(t.planes_accion) if t.planes_accion else []
        nueva_v = len(planes) + 1
        planes.append({
            "version": nueva_v,
            "plan": plan,
            "fecha": datetime.utcnow().isoformat()
        })
        t.planes_accion = planes
        t.status = "IN_PROGRESS"
        
        # Registrar en hilo
        db.add(TicketThread(
            id=str(uuid.uuid4()),
            ticket_id=ticket_id,
            author="SRE-Agent",
            content=f"**Fast-Track completado**\n\n**Veredicto:** {veredicto}\n\n**Plan V{nueva_v}:** {plan}"
        ))
        db.commit()
        db.close()
        
        return f"VEREDICTO: {veredicto}\n\nPLAN DE ACCION (V{nueva_v}): {plan}"
    
    except json.JSONDecodeError as e:
        db.close()
        return f"Error: El LLM no devolvio JSON valido. Respuesta raw: {res.content[:500]}"
    except Exception as e:
        db.close()
        return f"Error en Fast-Track: {str(e)}"


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
        ("system", """Eres AgentX, un Ingeniero SRE L1/L2 automatizado.
Técnicos disponibles: Alex SRE, Sonia DevOps, Carlos Cloud, Marta Security.

Tu flujo obligatorio:
1. Extraer gravedad y sistema afectado.
2. Usar 'buscar_conocimiento' para contextualizar con la base vectorial.
3. Crear tickets con veredictos y planes de acción.
4. Notificar al equipo y al usuario.

Si te dan una descripción de imagen/captura, analízala como evidencia técnica.
Si te dan un ID de ticket, usa 'leer_ticket' primero."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    return AgentExecutor(agent=create_tool_calling_agent(llm, tools, prompt), tools=tools, verbose=True)


def analyze_image_with_vision(image_b64: str, mime_type: str, user_text: str = "") -> str:
    """Usa GPT-4o Vision directamente para analizar una imagen y devuelve una descripción técnica en texto."""
    from langchain_core.messages import HumanMessage
    
    vision_llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=1500)
    
    content_blocks = []
    if user_text:
        content_blocks.append({"type": "text", "text": user_text})
    content_blocks.append({"type": "text", "text": "Analiza esta captura/imagen como un Ingeniero SRE. Describe exactamente qué ves: errores, logs, métricas, dashboards, stack traces, códigos de estado HTTP, etc. Sé extremadamente técnico y preciso."})
    content_blocks.append({
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}
    })
    
    msg = HumanMessage(content=content_blocks)
    response = vision_llm.invoke([msg])
    return response.content
