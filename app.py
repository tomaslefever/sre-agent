import os
import uuid
import base64
import streamlit as st
from datetime import datetime
from database import init_db, SessionLocal, Ticket, TicketThread, Attachment, Repository, ChatSession
from agent_engine import get_agent_executor, TECNICOS

# --- INICIALIZACIÓN ---
init_db()

st.set_page_config(page_title="AgentX: SRE & Ticketing", page_icon="🎫", layout="wide")
st.markdown("""<style>
    .block-container { padding-top: 1rem; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
    header[data-testid="stHeader"] { display: none; }
</style>""", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "seccion" not in st.session_state:
    st.session_state.seccion = "Centro de Incidentes"
if "selected_ticket" not in st.session_state:
    st.session_state.selected_ticket = None

with st.sidebar:
    st.title("🤖 AgentX SRE")
    if st.button("🕵️ Centro de Incidentes", use_container_width=True, type="primary" if st.session_state.seccion == "Centro de Incidentes" else "secondary"):
        st.session_state.seccion = "Centro de Incidentes"
        st.rerun()
    if st.button("📋 Tablero de Tickets", use_container_width=True, type="primary" if st.session_state.seccion == "Tablero de Tickets" else "secondary"):
        st.session_state.seccion = "Tablero de Tickets"
        st.rerun()
    if st.button("📚 Base de Conocimiento", use_container_width=True, type="primary" if st.session_state.seccion == "Base de Conocimiento" else "secondary"):
        st.session_state.seccion = "Base de Conocimiento"
        st.rerun()
    st.divider()
    if st.button("➕ Nueva Conversación", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.seccion = "Centro de Incidentes"
        st.rerun()

    # Sesiones pasadas
    st.caption("Conversaciones recientes")
    _db = SessionLocal()
    try:
        sessions = _db.query(ChatSession).order_by(ChatSession.created_at.desc()).limit(15).all()
        for s in sessions:
            label = f"{s.title[:30]}..." if len(s.title) > 30 else s.title
            is_active = st.session_state.session_id == s.id
            if st.button(label, key=f"ses_{s.id}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.session_id = s.id
                st.session_state.seccion = "Centro de Incidentes"
                st.rerun()
    finally:
        _db.close()

# --- SECCIÓN: CENTRO DE INCIDENTES ---
if st.session_state.seccion == "Centro de Incidentes":
    st.header("🕵️ Centro de Diagnóstico")
    from langchain_community.chat_message_histories import SQLChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from database import DB_URL

    agent_exec = get_agent_executor()
    agent_with_memory = RunnableWithMessageHistory(
        agent_exec,
        lambda id_s: SQLChatMessageHistory(session_id=id_s, connection_string=DB_URL),
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    # Mostrar tickets vinculados a esta sesión
    _sdb = SessionLocal()
    linked_tickets = _sdb.query(Ticket).filter(Ticket.session_id == st.session_state.session_id).all()
    if linked_tickets:
        for lt in linked_tickets:
            tc1, tc2 = st.columns([3, 1])
            tc1.info(f"🎫 **{lt.id}** — `{lt.status}` — {lt.assigned_to}")
            if tc2.button("Ver Ticket", key=f"go_{lt.id}", use_container_width=True):
                st.session_state.selected_ticket = lt.id
                st.session_state.seccion = "Tablero de Tickets"
                _sdb.close()
                st.rerun()
    _sdb.close()

    history = SQLChatMessageHistory(session_id=st.session_state.session_id, connection_string=DB_URL)
    for msg in history.messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    from streamlit_chat_prompt import prompt as chat_prompt

    response = chat_prompt(
        name="incident_chat",
        key="incident_chat",
        placeholder="Describe el incidente o pega una captura (Ctrl+V)...",
        main_bottom=True,
    )
    if response:
        text = response.text or ""
        images = response.images or []

        # Fase 1: Pre-procesar imágenes (ya vienen en base64)
        image_descriptions = []
        display_images = []

        for i, img in enumerate(images):
            img_name = f"imagen_{i+1}.png"
            img_b64 = img.data
            mime_type = img.type or "image/png"
            # Decodificar para mostrar en UI
            try:
                img_bytes = base64.b64decode(img_b64)
            except Exception:
                img_bytes = None
            if img_bytes:
                display_images.append((img_name, img_bytes, mime_type))
            try:
                from agent_engine import analyze_image_with_vision
                desc = analyze_image_with_vision(img_b64, mime_type, text)
                image_descriptions.append(f"[ANÁLISIS VISUAL de '{img_name}']:\n{desc}")
            except Exception as e:
                image_descriptions.append(f"[ERROR al analizar imagen '{img_name}': {str(e)}]")

        # Fase 2: Construir el input completo para el agente
        parts = []
        if text:
            parts.append(text)
        for desc in image_descriptions:
            parts.append(desc)
        if not parts:
            parts.append("El usuario ha adjuntado archivos sin mensaje adicional.")

        full_input = "\n\n".join(parts)

        # Mostrar en UI
        with st.chat_message("user"):
            if text:
                st.markdown(text)
            for name, img_bytes, mime in display_images:
                st.image(img_bytes, caption=f"📷 {name}")

        # Guardar/actualizar sesión
        _sdb2 = SessionLocal()
        existing_session = _sdb2.query(ChatSession).filter(ChatSession.id == st.session_state.session_id).first()
        if not existing_session:
            title = text[:50] if text else "Análisis de imagen"
            _sdb2.add(ChatSession(id=st.session_state.session_id, title=title, created_at=datetime.utcnow()))
            _sdb2.commit()
        _sdb2.close()

        # Fase 3: Enviar al agente
        with st.chat_message("assistant"):
            with st.spinner("AgentX analizando evidencia..."):
                res = agent_with_memory.invoke(
                    {"input": full_input},
                    config={"configurable": {"session_id": st.session_state.session_id}}
                )
                st.markdown(res["output"])


# --- SECCIÓN: TABLERO KANBAN ---
elif st.session_state.seccion == "Tablero de Tickets":
    st.header("📋 SRE Kanban Board")
    db = SessionLocal()
    try:
        if st.session_state.selected_ticket:
            t_id = st.session_state.selected_ticket
            t = db.query(Ticket).filter(Ticket.id == t_id).first()
            if st.button("⬅️ Regresar al Tablero"):
                st.session_state.selected_ticket = None
                st.rerun()
            if t:
                st.subheader(f"🎫 {t_id}")
                st.markdown(f"**Estado:** `{t.status}` | **Asignado:** `{t.assigned_to}`")
                c1, c2 = st.columns([2, 1])
                with c1:
                    with st.expander("📝 Reporte Original", expanded=True):
                        st.write(t.report)

                    # Adjuntos del incidente
                    adjuntos = db.query(Attachment).filter(Attachment.ticket_id == t_id).all()
                    if adjuntos:
                        with st.expander(f"📎 Adjuntos del Incidente ({len(adjuntos)})", expanded=False):
                            for adj in adjuntos:
                                if adj.file_data and adj.file_type and adj.file_type.startswith("image/"):
                                    st.image(adj.file_data, caption=adj.filename)
                                elif adj.file_data and adj.file_type and adj.file_type.startswith("text/"):
                                    try:
                                        st.code(adj.file_data.decode("utf-8"), language=None)
                                    except:
                                        st.caption(f"📄 {adj.filename}")
                                else:
                                    st.caption(f"📄 {adj.filename} ({adj.file_type or 'desconocido'})")
                                    if adj.file_data:
                                        st.download_button(
                                            f"Descargar {adj.filename}",
                                            data=adj.file_data,
                                            file_name=adj.filename,
                                            key=f"dl_{adj.id}"
                                        )

                    # Veredicto
                    if t.veredicto:
                        st.success(f"⚖️ **Veredicto:**\n\n{t.veredicto}")
                    else:
                        st.warning("⚠️ Sin veredicto aún. Ejecuta un **Fast-Track IA** para diagnosticar.")

                    # Planes de Acción con botones integrados
                    st.markdown("#### 🗺️ Planes de Acción")
                    planes = list(t.planes_accion) if t.planes_accion else []
                    if t.status != "AWAITING_VALIDATION":
                        if st.button("⚡ Generar Plan de Acción", use_container_width=True, type="primary"):
                            from agent_engine import diagnostico_fast_track
                            with st.spinner("🧠 Analizando con IA y generando plan..."):
                                resultado = diagnostico_fast_track.invoke({"ticket_id": t_id})
                            st.success("Plan generado")
                            st.info(resultado)
                            st.button("🔄 Recargar vista", key="reload_ft", on_click=lambda: st.rerun())
                    if planes:
                        for idx, p in enumerate(reversed(planes)):
                            v_num = p.get('version', '?')
                            with st.expander(f"Plan V{v_num} — {p.get('fecha', '')[:10]}", expanded=(idx == 0)):
                                archivos_rev = p.get("archivos_revisados", [])
                                hallazgos = p.get("hallazgos", [])
                                if archivos_rev:
                                    st.markdown(f"**📁 Archivos revisados ({len(archivos_rev)}):**")
                                    for a in archivos_rev:
                                        st.code(a, language=None)
                                if hallazgos:
                                    st.markdown(f"**🔍 Hallazgos ({len(hallazgos)}):**")
                                    for i, h in enumerate(hallazgos, 1):
                                        st.markdown(f"{i}. {h}")
                                st.markdown("**📋 Plan:**")
                                st.write(p.get("plan", "Sin detalle"))

                                # Botones de acción dentro del plan
                                if t.status == "AWAITING_VALIDATION":
                                    st.info("⏳ PR enviado — ticket en espera de validación.")
                                else:
                                    btn_c1, btn_c2 = st.columns(2)
                                    with btn_c1:
                                        if st.button(f"🚀 Ejecutar Plan V{v_num}", key=f"exec_plan_{v_num}", use_container_width=True, type="primary"):
                                            from agent_engine import ejecutar_plan_accion
                                            with st.spinner("🔧 Generando código en rama..."):
                                                res_exec = ejecutar_plan_accion.invoke({"ticket_id": t_id})
                                            if "Error" in res_exec:
                                                db.add(TicketThread(id=str(uuid.uuid4()), ticket_id=t_id, author="SRE-Agent", content=f"**Error al ejecutar plan:** {res_exec}"))
                                                db.commit()
                                            st.rerun()
                                    with btn_c2:
                                        if st.button(f"📬 Enviar PR V{v_num}", key=f"pr_plan_{v_num}", use_container_width=True):
                                            from agent_engine import crear_pr_ticket
                                            with st.spinner("📬 Creando Pull Request..."):
                                                res_pr = crear_pr_ticket.invoke({"ticket_id": t_id})
                                            if "Error" in res_pr:
                                                db.add(TicketThread(id=str(uuid.uuid4()), ticket_id=t_id, author="SRE-Agent", content=f"**Error al crear PR:** {res_pr}"))
                                                db.commit()
                                            st.rerun()

                with c2:
                    st.markdown("#### 💬 Historial")
                    hilos = db.query(TicketThread).filter(TicketThread.ticket_id == t_id).order_by(TicketThread.timestamp.asc()).all()
                    if not hilos:
                        st.info("Sin actividad aún.")
                    for h in hilos:
                        with st.chat_message("assistant" if h.author == "SRE-Agent" else "user"):
                            st.caption(f"{h.author} — {h.timestamp.strftime('%Y-%m-%d %H:%M')}")
                            st.write(h.content)
                    with st.form(key=f"form_hilo_{t_id}", clear_on_submit=True):
                        nuevo_txt = st.text_area("Añadir comentario...")
                        if st.form_submit_button("Enviar"):
                            if nuevo_txt:
                                db.add(TicketThread(id=str(uuid.uuid4()), ticket_id=t_id, author="SRE-Admin", content=nuevo_txt))
                                db.commit()
                                st.rerun()

            else:
                st.session_state.selected_ticket = None
        else:
            cols = st.columns(5)
            estados = [("Abierto", "ABIERTOS"), ("IN_PROGRESS", "PROGRESO"), ("PENDING_NOTIF", "REVISIÓN"), ("AWAITING_VALIDATION", "VALIDACIÓN"), ("RESOLVED", "RESUELTOS")]
            for i, (est_id, label) in enumerate(estados):
                with cols[i]:
                    st.markdown(f"### {label}")
                    tkts = db.query(Ticket).filter(Ticket.status == est_id).all()
                    for tk in tkts:
                        with st.container(border=True):
                            st.markdown(f"**{tk.id}**")
                            st.caption(f"👤 {tk.assigned_to}")
                            if st.button("📂 Ver Detalle", key=f"btn_{tk.id}", use_container_width=True):
                                st.session_state.selected_ticket = tk.id
                                st.rerun()
    finally:
        db.close()

# --- SECCIÓN: CONOCIMIENTO ---
elif st.session_state.seccion == "Base de Conocimiento":
    st.header("📚 GitHub Knowledge Base")
    db = SessionLocal()
    try:
        repos = db.query(Repository).all()
        for r in repos:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.write(f"🔗 **{r.url}**")
                c1.caption(f"Actualizado: {r.last_updated}")
                if c2.button("🔄 Actualizar", key=f"upd_{r.id}", use_container_width=True):
                    r.last_updated = datetime.utcnow()
                    db.commit()
                    st.toast("Actualización iniciada...")
                    st.rerun()
        st.divider()
        st.markdown("#### Vincular nuevo repositorio")
        url_input = st.text_input("GitHub URL")
        if st.button("📥 Sincronizar Nuevo", type="primary"):
            if url_input:
                new_r = Repository(id=str(uuid.uuid4()), url=url_input, last_updated=datetime.utcnow())
                db.add(new_r)
                db.commit()
                st.success("¡Vínculo exitoso!")
                st.rerun()
    finally:
        db.close()

    # --- DIAGNÓSTICO QDRANT ---
    st.divider()
    st.markdown("#### 🔬 Diagnóstico de Base Vectorial (Qdrant)")
    if st.button("🩺 Inspeccionar Qdrant", use_container_width=True):
        from agent_engine import diagnosticar_qdrant
        with st.spinner("Inspeccionando colección kb_sre..."):
            diag = diagnosticar_qdrant()

        if not diag["collection_exists"]:
            st.error("❌ La colección `kb_sre` NO existe en Qdrant.")
        elif diag["total_points"] == 0:
            st.warning("⚠️ La colección existe pero está VACÍA (0 puntos). No hay código indexado.")
        else:
            st.success(f"✅ Colección activa: **{diag['total_points']} vectores** indexados")

            archivos = diag.get("archivos", {})
            if archivos:
                st.markdown(f"**📁 Archivos indexados ({len(archivos)}):**")
                for nombre, count in sorted(archivos.items(), key=lambda x: -x[1]):
                    st.text(f"  {nombre} → {count} chunks")
            else:
                st.warning("No se encontraron nombres de archivo en los metadatos.")

            keys = diag.get("metadata_keys", [])
            if keys:
                st.markdown(f"**🔑 Claves de metadata:** `{', '.join(sorted(keys))}`")

            samples = diag.get("sample_payloads", [])
            if samples:
                with st.expander("🧪 Sample de payloads (primeros 3 puntos)", expanded=False):
                    for i, s in enumerate(samples):
                        st.json(s)

        if "error" in diag:
            st.error(f"Error: {diag['error']}")
