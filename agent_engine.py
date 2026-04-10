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


def diagnosticar_qdrant() -> dict:
    """Inspecciona Qdrant y devuelve un informe del estado de la base vectorial."""
    q_url = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
    client = QdrantClient(url=q_url)
    
    result = {
        "collection_exists": False,
        "total_points": 0,
        "archivos": {},
        "metadata_keys": set(),
        "sample_payloads": []
    }
    
    try:
        if not client.collection_exists("kb_sre"):
            return result
        
        result["collection_exists"] = True
        info = client.get_collection("kb_sre")
        result["total_points"] = info.points_count
        
        # Obtener sample de puntos para inspeccionar estructura
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
            # Recoger todas las claves de metadata
            result["metadata_keys"].update(payload.keys())
            if "metadata" in payload and isinstance(payload["metadata"], dict):
                result["metadata_keys"].update(f"metadata.{k}" for k in payload["metadata"].keys())
            
            # Extraer nombre de archivo
            src = (
                payload.get("source") 
                or payload.get("metadata", {}).get("source") 
                or "SIN_FUENTE"
            )
            result["archivos"][src] = result["archivos"].get(src, 0) + 1
        
        # Sample de los primeros 3 payloads para debug
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
    
    # 1. Buscar contexto con metadata
    v_store = get_vector_store()
    ctx = v_store.as_retriever(search_kwargs={"k": 10}).invoke(t.report)

    # Construir contexto CON metadata de archivo para que el LLM sepa la fuente
    context_parts = []
    for d in ctx:
        src = d.metadata.get("source", d.metadata.get("metadata", {}).get("source", "desconocido"))
        offset = d.metadata.get("start_index", "?")
        context_parts.append(f"[ARCHIVO: {src} | offset:{offset}]\n{d.page_content}")
    context_text = "\n---\n".join(context_parts) if context_parts else "Sin contexto disponible en la base de conocimiento."

    # 2. Prompt estructurado
    fast_llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)
    messages = [
        SystemMessage(content="Eres un Ingeniero SRE Senior experto en diagnostico rapido de incidentes. Siempre respondes en JSON valido."),
        HumanMessage(content=f"""Analiza este incidente usando UNICAMENTE el contexto del codigo/documentacion adjunto.

IMPORTANTE: Basa tu analisis SOLO en el codigo fuente proporcionado abajo. NO inventes archivos, logs ni informacion que no este en el contexto. Si un fragmento no es relevante al incidente, ignoralo.

INCIDENTE REPORTADO (TICKET {ticket_id}):
{t.report}

CONTEXTO DE LA BASE DE CONOCIMIENTO (codigo y documentacion):
{context_text}

Responde UNICAMENTE con un JSON valido con estas claves:
{{
  "veredicto": "Explicacion tecnica detallada de la causa raiz, referenciando SOLO archivos y secciones del codigo que aparecen en el contexto proporcionado",
  "archivos_revisados": ["lista de archivos del contexto que analizaste - usa los nombres exactos del campo ARCHIVO"],
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
    """Ejecuta el plan de accion del ticket: genera codigo corregido con IA y lo sube a una rama en GitHub. NO crea PR."""
    import json
    import requests
    from langchain_core.messages import SystemMessage, HumanMessage

    gh_token = os.getenv("GITHUB_TOKEN")
    if not gh_token:
        return "Error: GITHUB_TOKEN no configurado."

    db = SessionLocal()
    t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not t:
        db.close()
        return "Ticket no encontrado."

    planes = list(t.planes_accion) if t.planes_accion else []
    if not planes:
        db.close()
        return "No hay plan de accion. Ejecuta Fast-Track primero."

    ultimo_plan = planes[-1]
    plan_text = ultimo_plan.get("plan", "")
    archivos_rev = ultimo_plan.get("archivos_revisados", [])
    hallazgos = ultimo_plan.get("hallazgos", [])

    # Buscar repositorio vinculado
    from database import Repository
    repo = db.query(Repository).first()
    if not repo or not repo.url:
        db.close()
        return "No hay repositorio vinculado. Agrega uno en Base de Conocimiento."

    # Extraer owner/repo de la URL
    repo_url = repo.url.rstrip("/")
    parts = repo_url.replace("https://github.com/", "").replace("http://github.com/", "").split("/")
    if len(parts) < 2:
        db.close()
        return f"URL de repo invalida: {repo_url}"
    owner, repo_name = parts[0], parts[1].replace(".git", "")

    headers = {
        "Authorization": f"token {gh_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    api_base = f"https://api.github.com/repos/{owner}/{repo_name}"

    # Generar nombre de rama descriptivo con referencia al ticket
    import re
    veredicto_slug = ""
    if t.veredicto:
        # Tomar las primeras palabras del veredicto como descripcion
        slug = re.sub(r'[^a-z0-9\s]', '', t.veredicto[:60].lower())
        veredicto_slug = "-".join(slug.split()[:5])
    elif plan_text:
        slug = re.sub(r'[^a-z0-9\s]', '', plan_text[:60].lower())
        veredicto_slug = "-".join(slug.split()[:5])
    branch_name = f"fix/{ticket_id.lower()}/{veredicto_slug}" if veredicto_slug else f"fix/{ticket_id.lower()}"

    try:
        # 1. Obtener SHA de main
        ref_res = requests.get(f"{api_base}/git/ref/heads/main", headers=headers)
        if ref_res.status_code != 200:
            ref_res = requests.get(f"{api_base}/git/ref/heads/master", headers=headers)
        if ref_res.status_code != 200:
            db.close()
            return f"Error al obtener rama principal: {ref_res.json().get('message', ref_res.status_code)}"

        base_sha = ref_res.json()["object"]["sha"]

        # 2. Crear rama
        create_ref = requests.post(f"{api_base}/git/refs", headers=headers, json={
            "ref": f"refs/heads/{branch_name}",
            "sha": base_sha
        })
        if create_ref.status_code not in (200, 201, 422):  # 422 = ya existe
            db.close()
            return f"Error al crear rama: {create_ref.json().get('message', '')}"

        # 3. Para cada archivo mencionado, obtener codigo de Qdrant y generar fix
        v_store = get_vector_store()
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)

        archivos_modificados = []

        for archivo in archivos_rev:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            q_url = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
            client = QdrantClient(url=q_url)

            codigo_chunks = []
            for filter_key in ["metadata.source", "source"]:
                res_q = client.scroll(
                    collection_name="kb_sre",
                    scroll_filter=Filter(must=[FieldCondition(key=filter_key, match=MatchValue(value=archivo))]),
                    limit=50, with_payload=True
                )[0]
                if res_q:
                    codigo_chunks = [p.payload.get("page_content", p.payload.get("content", "")) for p in res_q]
                    break

            if not codigo_chunks:
                continue

            codigo_original = "\n".join(codigo_chunks)

            fix_messages = [
                SystemMessage(content="Eres un ingeniero de software senior. Tu tarea es aplicar correcciones al codigo fuente. Devuelve UNICAMENTE el codigo corregido completo, sin markdown, sin explicaciones."),
                HumanMessage(content=f"""ARCHIVO: {archivo}

CODIGO ORIGINAL:
{codigo_original}

PROBLEMA REPORTADO:
{t.report}

HALLAZGOS:
{chr(10).join(hallazgos) if hallazgos else 'Ver plan'}

PLAN DE CORRECCION:
{plan_text}

Devuelve el codigo corregido completo del archivo. Solo el codigo, sin backticks ni explicaciones.""")
            ]

            fix_res = llm.invoke(fix_messages)
            codigo_corregido = fix_res.content.strip()
            if codigo_corregido.startswith("```"):
                lines = codigo_corregido.split("\n")
                codigo_corregido = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            file_path = archivo.lstrip("/")
            file_res = requests.get(f"{api_base}/contents/{file_path}?ref={branch_name}", headers=headers)

            veredicto_corto = (t.veredicto[:80] + "...") if t.veredicto and len(t.veredicto) > 80 else (t.veredicto or "correccion automatica")
            commit_data = {
                "message": f"fix({ticket_id}): {file_path}\n\n{veredicto_corto}\n\nRef: {ticket_id}",
                "content": __import__("base64").b64encode(codigo_corregido.encode()).decode(),
                "branch": branch_name
            }
            if file_res.status_code == 200:
                commit_data["sha"] = file_res.json()["sha"]

            put_res = requests.put(f"{api_base}/contents/{file_path}", headers=headers, json=commit_data)
            if put_res.status_code in (200, 201):
                archivos_modificados.append(file_path)

        # Registrar en ticket (sin crear PR)
        t.status = "PENDING_NOTIF"
        db.add(TicketThread(
            id=str(uuid.uuid4()), ticket_id=ticket_id, author="SRE-Agent",
            content=f"**Codigo generado en rama `{branch_name}`**\n\nArchivos modificados: {', '.join(archivos_modificados)}"
        ))
        db.commit()
        db.close()
        return f"Codigo subido a rama {branch_name}\nArchivos: {', '.join(archivos_modificados)}\n\nUsa 'Enviar PR' para crear el Pull Request."

    except Exception as e:
        t.status = "PENDING_NOTIF"
        db.add(TicketThread(
            id=str(uuid.uuid4()), ticket_id=ticket_id, author="SRE-Agent",
            content=f"Error al ejecutar plan: {str(e)}"
        ))
        db.commit()
        db.close()
        return f"Error: {str(e)}"


@tool
def crear_pr_ticket(ticket_id: str) -> str:
    """Crea un Pull Request en GitHub para la rama de fix de un ticket. Debe ejecutarse despues de ejecutar_plan_accion."""
    import requests

    gh_token = os.getenv("GITHUB_TOKEN")
    if not gh_token:
        return "Error: GITHUB_TOKEN no configurado."

    db = SessionLocal()
    t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not t:
        db.close()
        return "Ticket no encontrado."

    from database import Repository
    repo = db.query(Repository).first()
    if not repo or not repo.url:
        db.close()
        return "No hay repositorio vinculado."

    repo_url = repo.url.rstrip("/")
    parts = repo_url.replace("https://github.com/", "").replace("http://github.com/", "").split("/")
    if len(parts) < 2:
        db.close()
        return f"URL de repo invalida: {repo_url}"
    owner, repo_name = parts[0], parts[1].replace(".git", "")

    headers = {
        "Authorization": f"token {gh_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    api_base = f"https://api.github.com/repos/{owner}/{repo_name}"
    branch_prefix = f"fix/{ticket_id.lower()}"

    # Buscar rama del ticket (puede tener slug descriptivo)
    refs_res = requests.get(f"{api_base}/git/matching-refs/heads/{branch_prefix}", headers=headers)
    if refs_res.status_code != 200 or not refs_res.json():
        db.close()
        return f"No se encontro rama para {ticket_id}. Ejecuta el plan primero."
    branch_name = refs_res.json()[-1]["ref"].replace("refs/heads/", "")

    # Detectar rama base
    base_check = requests.get(f"{api_base}/git/ref/heads/main", headers=headers)
    base_branch = "main" if base_check.status_code == 200 else "master"

    planes = list(t.planes_accion) if t.planes_accion else []
    plan_text = planes[-1].get("plan", "N/A") if planes else "N/A"

    # Obtener archivos modificados del hilo
    hilos = db.query(TicketThread).filter(TicketThread.ticket_id == ticket_id).all()
    archivos_info = ""
    for h in hilos:
        if "Archivos modificados:" in h.content:
            archivos_info = h.content.split("Archivos modificados:")[-1].strip()
            break

    pr_body = f"""## Ticket: {ticket_id}

### Reporte
{t.report[:500]}

### Veredicto
{t.veredicto or 'N/A'}

### Archivos modificados
{archivos_info or 'Ver commits en la rama'}

### Plan aplicado
{plan_text}

---
*Generado automaticamente por AgentX SRE*"""

    try:
        pr_res = requests.post(f"{api_base}/pulls", headers=headers, json={
            "title": f"fix({ticket_id}): {t.veredicto[:80] if t.veredicto else 'Correccion automatica'}",
            "body": pr_body,
            "head": branch_name,
            "base": base_branch
        })

        if pr_res.status_code in (200, 201):
            pr_url = pr_res.json().get("html_url", "")
            t.status = "AWAITING_VALIDATION"
            db.add(TicketThread(
                id=str(uuid.uuid4()), ticket_id=ticket_id, author="SRE-Agent",
                content=f"**PR creado:** [{branch_name}]({pr_url})\n\nTicket en espera de validacion."
            ))
            db.commit()
            db.close()
            return f"PR creado exitosamente: {pr_url}"
        else:
            err = pr_res.json().get("message", str(pr_res.status_code))
            db.add(TicketThread(
                id=str(uuid.uuid4()), ticket_id=ticket_id, author="SRE-Agent",
                content=f"**Error al crear PR:** {err}\n\nRama: `{branch_name}`"
            ))
            db.commit()
            db.close()
            return f"Error al crear PR: {err}"

    except Exception as e:
        db.add(TicketThread(
            id=str(uuid.uuid4()), ticket_id=ticket_id, author="SRE-Agent",
            content=f"**Error al crear PR:** {str(e)}"
        ))
        db.commit()
        db.close()
        return f"Error: {str(e)}"

tools = [buscar_conocimiento, listar_archivos_conocimiento, leer_archivo_conocimiento, buscar_codigo_detallado, leer_ticket, crear_ticket_sre, actualizar_veredicto, generar_plan_accion, diagnostico_fast_track, ejecutar_plan_accion, crear_pr_ticket]

def get_agent_executor():
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres AgentX, un Ingeniero SRE L1/L2 automatizado.
Tecnicos disponibles: Alex SRE, Sonia DevOps, Carlos Cloud, Marta Security.

Tu flujo obligatorio:
1. Extraer gravedad y sistema afectado del reporte.
2. SIEMPRE usar 'listar_archivos_conocimiento' para ver que codigo tienes disponible.
3. Usar 'buscar_conocimiento' para una primera busqueda rapida.
4. SIEMPRE profundizar con 'buscar_codigo_detallado' (limite=20 o mas) para obtener contexto completo.
5. Usar 'leer_archivo_conocimiento' para leer archivos completos sospechosos. Lee TODOS los archivos relevantes sin importar cuantos sean.
6. Crear tickets con veredictos detallados y planes de accion.

REGLAS DE BUSQUEDA (CRITICO):
- NUNCA te conformes con una sola busqueda. Haz multiples busquedas con diferentes terminos.
- Si encuentras archivos relevantes, LEELOS COMPLETOS con 'leer_archivo_conocimiento'.
- Busca exhaustivamente sin importar cuanto tarde. La precision es mas importante que la velocidad.
- Si un resultado menciona otros archivos, busca y lee esos archivos tambien.

REGLAS DE INFORME:
- SIEMPRE referencia los archivos que revisaste por nombre.
- SIEMPRE indica las secciones/lineas/offsets donde encontraste problemas.
- Genera un informe estructurado: Archivos Revisados -> Hallazgos -> Veredicto -> Plan.

Si te dan una descripcion de imagen/captura, analiza la evidencia visual.
Si te dan un ID de ticket, usa 'leer_ticket' primero."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    return AgentExecutor(agent=create_tool_calling_agent(llm, tools, prompt), tools=tools, verbose=True)


def get_ticket_agent(ticket_id: str, ticket_report: str, attachments_text: str = ""):
    """Crea un agente conversacional contextualizado para un ticket especifico."""
    ticket_tools = [buscar_conocimiento, listar_archivos_conocimiento, leer_archivo_conocimiento, buscar_codigo_detallado, leer_ticket]
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Eres AgentX, asistente SRE experto asignado al ticket {ticket_id}.

CONTEXTO DEL INCIDENTE:
{ticket_report}

{f"ARCHIVOS ADJUNTOS DEL INCIDENTE:{chr(10)}{attachments_text}" if attachments_text else ""}

Tienes acceso a la base de conocimiento vectorial para buscar y leer codigo fuente.
Responde preguntas sobre el incidente, analiza codigo relacionado, y ayuda a investigar la causa raiz.
Busca exhaustivamente en la base vectorial sin importar cuanto tarde. Lee archivos completos cuando sea necesario."""),
        ("placeholder", "{{chat_history}}"),
        ("human", "{{input}}"),
        ("placeholder", "{{agent_scratchpad}}"),
    ])
    return AgentExecutor(agent=create_tool_calling_agent(llm, ticket_tools, prompt), tools=ticket_tools, verbose=True, max_iterations=25)


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
