import os
import uuid
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

# Sidebar
with st.sidebar:
    st.title("🤖 AgentX SRE")
    st.session_state.seccion = st.radio("Navegación", ["Centro de Incidentes", "Tablero de Tickets", "Base de Conocimiento"])
    st.divider()
    if st.button("🗑️ Limpiar Chat"):
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# --- SECCIÓN: CENTRO DE INCIDENTES ---
if st.session_state.seccion == "Centro de Incidentes":
    st.header("🕵️ Centro de Diagnóstico")
    
    # Historial de Chat
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

    # Mostrar mensajes previos
    history = SQLChatMessageHistory(session_id=st.session_state.session_id, connection_string=DB_URL)
    for msg in history.messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # Input Multimodal (Streamlit 1.39+)
    u_input = st.chat_input("Describe el incidente o pega un log...", accept_file=True)
    if u_input:
        f = u_input.get("file")
        text = u_input.get("text", "")
        
        # Lógica de procesamiento de archivos (simplificada aquí para brevedad)
        # ... (podría moverse a agent_engine también)
        
        with st.chat_message("user"):
            st.markdown(text)
        
        with st.chat_message("assistant"):
            with st.spinner("AgentX analizando..."):
                res = agent_with_memory.invoke(
                    {"input": text}, 
                    config={"configurable": {"session_id": st.session_state.session_id}}
                )
                st.markdown(res["output"])

# --- SECCIÓN: TABLERO KANBAN ---
elif st.session_state.seccion == "Tablero de Tickets":
    st.header("📋 SRE Kanban Board")
    db = SessionLocal()
    
    if st.session_state.selected_ticket:
        # VISTA DETALLE
        t_id = st.session_state.selected_ticket
        t = db.query(Ticket).filter(Ticket.id == t_id).first()
        if st.button("⬅️ Regresar"):
            st.session_state.selected_ticket = None
            st.rerun()
            
        if t:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader(f"Ticket: {t_id}")
                st.info(f"**Reporte:**\n{t.report}")
                st.divider()
                st.markdown("#### 💬 Hilo de Seguimiento")
                threads = db.query(TicketThread).filter(TicketThread.ticket_id == t_id).order_by(TicketThread.timestamp.asc()).all()
                for th in threads:
                    st.caption(f"**{th.author}** @ {th.timestamp.strftime('%H:%M')}")
                    st.write(th.content)
                
                comment = st.text_area("Añadir actualización...")
                if st.button("Comentar"):
                    db.add(TicketThread(id=str(uuid.uuid4()), ticket_id=t_id, author="Usuario", content=comment))
                    db.commit()
                    st.rerun()
            with col2:
                st.markdown("#### ⚡ Acciones Rápidas")
                if st.button("🚀 Fast-Track IA", use_container_width=True):
                    st.info("Disparando análisis rápido...")
                    # Aquí llamaríamos a la misma lógica de la herramienta
                if st.button("✅ Resolver", use_container_width=True, type="primary"):
                    t.status = "RESOLVED"
                    db.commit()
                    st.success("Ticket Resuelto!")
                    st.rerun()
    else:
        # VISTA KANBAN
        cols = st.columns(4)
        estados = [("Abierto", "ABIERTOS"), ("IN_PROGRESS", "PROGRESO"), ("PENDING_NOTIF", "REVISIÓN"), ("RESOLVED", "RESUELTOS")]
        
        for i, (est_id, label) in enumerate(estados):
            with cols[i]:
                st.markdown(f"**{label}**")
                tkts = db.query(Ticket).filter(Ticket.status == est_id).all()
                for tk in tkts:
                    with st.container(border=True):
                        st.caption(tk.id)
                        st.write(f"👤 {tk.assigned_to}")
                        if st.button("👁️ Detalle", key=tk.id):
                            st.session_state.selected_ticket = tk.id
                            st.rerun()
    db.close()

# --- SECCIÓN: CONOCIMIENTO ---
elif st.session_state.seccion == "Base de Conocimiento":
    st.header("📚 GitHub Knowledge Base")
    db = SessionLocal()
    repos = db.query(Repository).all()
    for r in repos:
        st.write(f"🔗 {r.url} (Actualizado: {r.last_updated})")
    
    new_repo = st.text_input("Agregar Repo GitHub (URL)")
    if st.button("Sincronizar"):
        # Lógica de sincronización
        st.success("Repo agregado!")
    db.close()