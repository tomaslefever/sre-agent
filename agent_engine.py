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
    """Busca en la base de conocimiento vectorial. Devuelve fragmentos con archivo fuente y lineas."""
    v_store = get_vector_store()
    docs = v_store.as_retriever(search_kwargs={"k": 8}).invoke(query)
    resultados = []
    for d in docs:
        src = d.metadata.get("source", "desconocido")
        start = d.metadata.get("start_index", "?")
        chunk_id = d.metadata.get("chunk_id", "?")
        resultados.append(f"[ARCHIVO: {src} | chunk:{chunk_id} | offset:{start}]\n{d.page_content}")
    return "\n\n---\n\n".join(resultados) if resultados else "No se encontro nada."

@tool
def listar_archivos_conocimiento(repo_filtro: str = None) -> str:
    """Lista TODOS los archivos unicos indexados en Qdrant. Usa esto para saber que codigo tienes disponible antes de leer."""
    q_url = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
    client = QdrantClient(url=q_url)
    all_points = []
    offset = None
    # Paginar para obtener todos los archivos
    while True:
        batch, next_offset = client.scroll(collection_name="kb_sre", limit=100, with_payload=True, offset=offset)
        all_points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset
    archivos = {}
    for p in all_points:
        src = p.payload.get("source", p.payload.get("metadata", {}).get("source", "desconocido"))
        if repo_filtro and repo_filtro not in src:
            continue
        if src not in archivos:
            archivos[src] = 0
        archivos[src] += 1
    lines = [f"- {name} ({count} chunks)" for name, count in sorted(archivos.items())]
    return f"Total: {len(archivos)} archivos indexados\n" + "\n".join(lines)

@tool
def leer_archivo_conocimiento(nombre_archivo: str) -> str:
    """Lee TODOS los chunks de un archivo especifico. Devuelve el contenido completo reconstruido con marcadores de posicion."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    q_url = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
    client = QdrantClient(url=q_url)
    
    # Intentar con metadata.source y con source directo
    for filter_key in ["metadata.source", "source"]:
        res = client.scroll(
            collection_name="kb_sre",
            scroll_filter=Filter(must=[FieldCondition(key=filter_key, match=MatchValue(value=nombre_archivo))]),
            limit=50,
            with_payload=True
        )[0]
        if res:
            break
    
    if not res:
        return f"Archivo '{nombre_archivo}' no encontrado. Usa 'listar_archivos_conocimiento' para ver archivos disponibles."
    
    # Reconstruir con marcadores de posicion
    chunks = []
    for i, p in enumerate(res):
        content = p.payload.get("page_content", p.payload.get("content", ""))
        offset = p.payload.get("start_index", p.payload.get("metadata", {}).get("start_index", "?"))
        chunks.append(f"--- [Chunk {i+1}/{len(res)} | offset:{offset}] ---\n{content}")
    
    return f"ARCHIVO: {nombre_archivo} ({len(res)} fragmentos)\n\n" + "\n\n".join(chunks)

@tool
def buscar_codigo_detallado(query: str, limite: int = 20) -> str:
    """Busqueda profunda en el codigo: Obtiene hasta 20 fragmentos relevantes con metadata completa (archivo, posicion). Usa esto cuando necesites mas contexto que buscar_conocimiento."""
    v_store = get_vector_store()
    k = min(limite, 30)
    docs = v_store.as_retriever(search_kwargs={"k": k}).invoke(query)
    
    # Agrupar por archivo
    por_archivo = {}
    for d in docs:
        src = d.metadata.get("source", d.metadata.get("metadata", {}).get("source", "desconocido"))
        offset = d.metadata.get("start_index", "?")
        if src not in por_archivo:
            por_archivo[src] = []
        por_archivo[src].append({"offset": offset, "content": d.page_content})
    
    informe = []
    for archivo, fragments in por_archivo.items():
        informe.append(f"\n### ARCHIVO: {archivo} ({len(fragments)} coincidencias)")
        for f in fragments:
            informe.append(f"  [offset:{f['offset']}]\n{f['content']}")
    
    return f"Busqueda profunda: {len(docs)} resultados en {len(por_archivo)} archivos\n" + "\n---\n".join(informe)

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
    fast_llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)
    messages = [
        SystemMessage(content="Eres un Ingeniero SRE Senior experto en diagnostico rapido de incidentes. Siempre respondes en JSON valido."),
        HumanMessage(content=f"""Analiza este incidente usando el contexto del codigo/documentacion adjunto.

INCIDENTE REPORTADO:
{t.report}

CONTEXTO DE LA BASE DE CONOCIMIENTO (codigo y documentacion):
{context_text}

Responde UNICAMENTE con un JSON valido con estas claves:
{{
  "veredicto": "Explicacion tecnica detallada de la causa raiz, referenciando archivos y secciones del codigo donde identificaste el problema",
  "archivos_revisados": ["lista de archivos que analizaste del contexto"],
  "hallazgos": ["lista de hallazgos especificos con referencia al archivo y fragmento de codigo donde encontraste cada problema"],
  "plan": "Pasos numerados y concretos para resolver el incidente, indicando que archivos modificar y que cambios hacer"
}}""")
    ]
    
    try:
        res = fast_llm.invoke(messages)
        raw = res.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        
        veredicto = data.get("veredicto", "No se pudo determinar la causa raiz.")
        plan = data.get("plan", "No se pudo generar un plan.")
        archivos_rev = data.get("archivos_revisados", [])
        hallazgos = data.get("hallazgos", [])
        
        # Construir informe completo
        informe = f"**Veredicto:** {veredicto}\n\n"
        if archivos_rev:
            informe += f"**Archivos revisados ({len(archivos_rev)}):**\n"
            for a in archivos_rev:
                informe += f"- `{a}`\n"
        if hallazgos:
            informe += f"\n**Hallazgos ({len(hallazgos)}):**\n"
            for i, h in enumerate(hallazgos, 1):
                informe += f"{i}. {h}\n"
        informe += f"\n**Plan de Accion:** {plan}"
        
        # Guardar en DB
        t.veredicto = veredicto
        planes = list(t.planes_accion) if t.planes_accion else []
        nueva_v = len(planes) + 1
        planes.append({
            "version": nueva_v,
            "plan": plan,
            "archivos_revisados": archivos_rev,
            "hallazgos": hallazgos,
            "fecha": datetime.utcnow().isoformat()
        })
        t.planes_accion = planes
        t.status = "IN_PROGRESS"
        
        # Registrar en hilo
        db.add(TicketThread(
            id=str(uuid.uuid4()),
            ticket_id=ticket_id,
            author="SRE-Agent",
            content=f"**Fast-Track completado**\n\n{informe}"
        ))
        db.commit()
        db.close()
        
        return informe
    
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

tools = [buscar_conocimiento, listar_archivos_conocimiento, leer_archivo_conocimiento, buscar_codigo_detallado, leer_ticket, crear_ticket_sre, actualizar_veredicto, generar_plan_accion, diagnostico_fast_track, ejecutar_plan_accion]

def get_agent_executor():
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres AgentX, un Ingeniero SRE L1/L2 automatizado.
Tecnicos disponibles: Alex SRE, Sonia DevOps, Carlos Cloud, Marta Security.

Tu flujo obligatorio:
1. Extraer gravedad y sistema afectado del reporte.
2. SIEMPRE usar 'listar_archivos_conocimiento' para ver que codigo tienes.
3. Usar 'buscar_conocimiento' o 'buscar_codigo_detallado' para encontrar codigo relevante.
4. Usar 'leer_archivo_conocimiento' para leer archivos completos sospechosos.
5. Crear tickets con veredictos detallados y planes de accion.

REGLAS DE INFORME:
- SIEMPRE referencia los archivos que revisaste por nombre.
- SIEMPRE indica las secciones/lineas/offsets donde encontraste problemas.
- Si un primer busqueda no da suficiente contexto, usa 'buscar_codigo_detallado' con mas chunks.
- Genera un informe estructurado: Archivos Revisados -> Hallazgos -> Veredicto -> Plan.

Si te dan una descripcion de imagen/captura, analiza la evidencia visual.
Si te dan un ID de ticket, usa 'leer_ticket' primero."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    return AgentExecutor(agent=create_tool_calling_agent(llm, tools, prompt), tools=tools, verbose=True)


def analyze_image_with_vision(image_b64: str, mime_type: str, user_text: str = "") -> str:
    """Usa GPT-4o Vision directamente para analizar una imagen y devuelve una descripción técnica en texto."""
    from langchain_core.messages import HumanMessage
    
    vision_llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0, max_tokens=1500)
    
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
