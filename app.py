import os
import uuid
import base64
import streamlit as st
from datetime import datetime
from database import init_db, SessionLocal, Ticket, TicketThread, Repository
from agent_engine import get_agent_executor, TECNICOS

# --- INICIALIZACIÓN ---
init_db()

st.set_page_config(page_title="AgentX: SRE & Ticketing", page_icon="🎫", layout="wide")
st.markdown('<script src="https://cdn.tailwindcss.com"></script>', unsafe_allow_html=True)

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
    if st.button("🗑️ Limpiar Chat"):
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

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

    history = SQLChatMessageHistory(session_id=st.session_state.session_id, connection_string=DB_URL)
    for msg in history.messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    u_input = st.chat_input(
        "Describe el incidente o pega una captura (Ctrl+V)...",
        accept_file="multiple",
        file_type=["png", "jpg", "jpeg", "gif", "webp", "txt", "log", "csv", "json"]
    )
    if u_input:
        text = u_input.text or ""
        uploaded_files = u_input.files or []
        
        # Fase 1: Pre-procesar archivos
        image_descriptions = []
        file_texts = []
        display_images = []
        
        for f in uploaded_files:
            f_bytes = f.read()
            if f.type and f.type.startswith("image/"):
                # Convertir a base64 y analizar con GPT-4o Vision
                img_b64 = base64.b64encode(f_bytes).decode()
                display_images.append((f.name, f_bytes, f.type))
                try:
                    from agent_engine import analyze_image_with_vision
                    desc = analyze_image_with_vision(img_b64, f.type, text)
                    image_descriptions.append(f"[ANÁLISIS VISUAL de '{f.name}']:\n{desc}")
                except Exception as e:
                    image_descriptions.append(f"[ERROR al analizar imagen '{f.name}': {str(e)}]")
            else:
                # Archivos de texto/log
                try:
                    txt = f_bytes.decode("utf-8")
                    file_texts.append(f"[CONTENIDO de '{f.name}']:\n{txt}")
                except:
                    file_texts.append(f"[ARCHIVO '{f.name}' adjunto (binario, no legible)]")
        
        # Fase 2: Construir el input completo para el agente (siempre texto)
        parts = []
        if text:
            parts.append(text)
        for desc in image_descriptions:
            parts.append(desc)
        for ft in file_texts:
            parts.append(ft)
        if not parts:
            parts.append("El usuario ha adjuntado archivos sin mensaje adicional.")
        
        full_input = "\n\n".join(parts)
        
        # Mostrar en UI
        with st.chat_message("user"):
            if text:
                st.markdown(text)
            for name, img_bytes, mime in display_images:
                st.image(img_bytes, caption=f"📷 {name}")
            for ft in file_texts:
                st.caption(f"📎 Archivo adjunto")
        
        # Fase 3: Enviar al agente (siempre string, con la descripción de visión incluida)
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
                    if t.veredicto:
                        st.success(f"⚖️ **Veredicto:**\n{t.veredicto}")
                    st.divider()
                    st.markdown("#### 💬 Historial y Comentarios")
                    hilos = db.query(TicketThread).filter(TicketThread.ticket_id == t_id).order_by(TicketThread.timestamp.asc()).all()
                    for h in hilos:
                        with st.chat_message("assistant" if h.author == "SRE-Agent" else "user"):
                            st.caption(f"{h.author} - {h.timestamp.strftime('%Y-%m-%d %H:%M')}")
                            st.write(h.content)
                    with st.form(key=f"form_hilo_{t_id}", clear_on_submit=True):
                        nuevo_txt = st.text_area("Añadir comentario...")
                        if st.form_submit_button("Enviar"):
                            if nuevo_txt:
                                db.add(TicketThread(id=str(uuid.uuid4()), ticket_id=t_id, author="SRE-Admin", content=nuevo_txt))
                                db.commit()
                                st.rerun()
                with c2:
                    st.markdown("#### ⚙️ Gestión de Incidente")
                    if st.button("⚡ Fast-Track IA", use_container_width=True):
                        from agent_engine import diagnostico_fast_track
                        res = diagnostico_fast_track.invoke({"ticket_id": t_id})
                        st.toast(res)
                        st.rerun()
                    if st.button("🚀 Ejecutar Plan", use_container_width=True, type="primary"):
                        from agent_engine import ejecutar_plan_accion
                        res = ejecutar_plan_accion.invoke({"ticket_id": t_id})
                        st.toast(res)
                        st.rerun()
            else:
                st.session_state.selected_ticket = None
        else:
            cols = st.columns(4)
            estados = [("Abierto", "ABIERTOS"), ("IN_PROGRESS", "PROGRESO"), ("PENDING_NOTIF", "REVISIÓN"), ("RESOLVED", "RESUELTOS")]
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