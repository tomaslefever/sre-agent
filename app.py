import os
import uuid
import base64
import streamlit as st
from datetime import datetime
from database import init_db, SessionLocal, Ticket, TicketThread, Attachment, Repository, ChatSession
from agent_engine import get_agent_executor, TECHNICIANS

# --- INIT ---
init_db()

st.set_page_config(page_title="AgentX: SRE & Ticketing", page_icon="🎫", layout="wide")
st.markdown("""<style>
    .block-container { padding-top: 1rem; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
    header[data-testid="stHeader"] { display: none; }
    textarea { max-width: 100% !important; box-sizing: border-box !important; }
    .stTextArea > div > div > textarea { width: 100% !important; }
    iframe { max-width: 100% !important; }
</style>""", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "section" not in st.session_state:
    st.session_state.section = "Incident Center"
if "selected_ticket" not in st.session_state:
    st.session_state.selected_ticket = None

with st.sidebar:
    st.title("🤖 AgentX SRE")
    if st.button("🕵️ Incident Center", use_container_width=True, type="primary" if st.session_state.section == "Incident Center" else "secondary"):
        st.session_state.section = "Incident Center"
        st.rerun()
    if st.button("📋 Ticket Board", use_container_width=True, type="primary" if st.session_state.section == "Ticket Board" else "secondary"):
        st.session_state.section = "Ticket Board"
        st.rerun()
    if st.button("📚 Knowledge Base", use_container_width=True, type="primary" if st.session_state.section == "Knowledge Base" else "secondary"):
        st.session_state.section = "Knowledge Base"
        st.rerun()
    st.divider()
    if st.button("➕ New Conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.section = "Incident Center"
        st.rerun()

    # Past sessions
    st.caption("Recent conversations")
    _db = SessionLocal()
    try:
        sessions = _db.query(ChatSession).order_by(ChatSession.created_at.desc()).limit(15).all()
        for s in sessions:
            label = f"{s.title[:30]}..." if len(s.title) > 30 else s.title
            is_active = st.session_state.session_id == s.id
            if st.button(label, key=f"ses_{s.id}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.session_id = s.id
                st.session_state.section = "Incident Center"
                st.rerun()
    finally:
        _db.close()

# --- SECTION: INCIDENT CENTER ---
if st.session_state.section == "Incident Center":
    st.header("🕵️ Diagnostic Center")
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

    # Show tickets linked to this session
    _sdb = SessionLocal()
    linked_tickets = _sdb.query(Ticket).filter(Ticket.session_id == st.session_state.session_id).all()
    if linked_tickets:
        for lt in linked_tickets:
            tc1, tc2 = st.columns([3, 1])
            tc1.info(f"🎫 **{lt.id}** — `{lt.status}` — {lt.assigned_to}")
            if tc2.button("View Ticket", key=f"go_{lt.id}", use_container_width=True):
                st.session_state.selected_ticket = lt.id
                st.session_state.section = "Ticket Board"
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
        placeholder="Describe the incident or paste a screenshot (Ctrl+V)...",
        main_bottom=True,
    )
    if response:
        text = response.text or ""
        images = response.images or []

        # Phase 1: Pre-process images (already base64)
        image_descriptions = []
        display_images = []

        for i, img in enumerate(images):
            img_name = f"image_{i+1}.png"
            img_b64 = img.data
            mime_type = img.type or "image/png"
            try:
                img_bytes = base64.b64decode(img_b64)
            except Exception:
                img_bytes = None
            if img_bytes:
                display_images.append((img_name, img_bytes, mime_type))
            try:
                from agent_engine import analyze_image_with_vision
                desc = analyze_image_with_vision(img_b64, mime_type, text)
                image_descriptions.append(f"[VISUAL ANALYSIS of '{img_name}']:\n{desc}")
            except Exception as e:
                image_descriptions.append(f"[ERROR analyzing image '{img_name}': {str(e)}]")

        # Phase 2: Build full input for the agent
        parts = []
        if text:
            parts.append(text)
        for desc in image_descriptions:
            parts.append(desc)
        if not parts:
            parts.append("The user attached files without additional message.")

        full_input = "\n\n".join(parts)

        # Show in UI
        with st.chat_message("user"):
            if text:
                st.markdown(text)
            for name, img_bytes, mime in display_images:
                st.image(img_bytes, caption=f"📷 {name}")

        # Save/update session
        _sdb2 = SessionLocal()
        existing_session = _sdb2.query(ChatSession).filter(ChatSession.id == st.session_state.session_id).first()
        if not existing_session:
            title = text[:50] if text else "Image analysis"
            _sdb2.add(ChatSession(id=st.session_state.session_id, title=title, created_at=datetime.utcnow()))
            _sdb2.commit()
        _sdb2.close()

        # Phase 3: Send to agent
        with st.chat_message("assistant"):
            with st.spinner("AgentX analyzing evidence..."):
                res = agent_with_memory.invoke(
                    {"input": full_input},
                    config={"configurable": {"session_id": st.session_state.session_id}}
                )
                st.markdown(res["output"])


# --- SECTION: KANBAN BOARD ---
elif st.session_state.section == "Ticket Board":
    st.header("📋 SRE Kanban Board")
    db = SessionLocal()
    try:
        if st.session_state.selected_ticket:
            t_id = st.session_state.selected_ticket
            t = db.query(Ticket).filter(Ticket.id == t_id).first()
            if st.button("⬅️ Back to Board"):
                st.session_state.selected_ticket = None
                st.rerun()
            if t:
                st.subheader(f"🎫 {t_id}")
                st.markdown(f"**Status:** `{t.status}` | **Assigned:** `{t.assigned_to}`")
                c1, c2 = st.columns([2, 1])
                with c1:
                    with st.expander("📝 Original Report", expanded=True):
                        st.write(t.report)

                    # Incident attachments
                    attachments = db.query(Attachment).filter(Attachment.ticket_id == t_id).all()
                    if attachments:
                        with st.expander(f"📎 Incident Attachments ({len(attachments)})", expanded=False):
                            for adj in attachments:
                                if adj.file_data and adj.file_type and adj.file_type.startswith("image/"):
                                    st.image(adj.file_data, caption=adj.filename)
                                elif adj.file_data and adj.file_type and adj.file_type.startswith("text/"):
                                    try:
                                        st.code(adj.file_data.decode("utf-8"), language=None)
                                    except:
                                        st.caption(f"📄 {adj.filename}")
                                else:
                                    st.caption(f"📄 {adj.filename} ({adj.file_type or 'unknown'})")
                                    if adj.file_data:
                                        st.download_button(
                                            f"Download {adj.filename}",
                                            data=adj.file_data,
                                            file_name=adj.filename,
                                            key=f"dl_{adj.id}"
                                        )

                    # Verdict
                    if t.veredicto:
                        st.success(f"⚖️ **Verdict:**\n\n{t.veredicto}")
                    else:
                        st.warning("⚠️ No verdict yet. Generate an **Action Plan** to diagnose.")

                    # Action Plans with integrated buttons
                    st.markdown("#### 🗺️ Action Plans")
                    planes = list(t.planes_accion) if t.planes_accion else []
                    if t.status != "AWAITING_VALIDATION":
                        if st.button("⚡ Generate Action Plan", use_container_width=True, type="primary"):
                            from agent_engine import diagnostico_fast_track
                            with st.spinner("🧠 Analyzing with AI and generating plan..."):
                                resultado = diagnostico_fast_track.invoke({"ticket_id": t_id})
                            st.success("Plan generated")
                            st.info(resultado)
                            st.button("🔄 Reload view", key="reload_ft", on_click=lambda: st.rerun())
                    if planes:
                        for idx, p in enumerate(reversed(planes)):
                            v_num = p.get('version', '?')
                            with st.expander(f"Plan V{v_num} — {p.get('fecha', '')[:10]}", expanded=(idx == 0)):
                                archivos_rev = p.get("archivos_revisados", [])
                                hallazgos = p.get("hallazgos", [])
                                if archivos_rev:
                                    st.markdown(f"**📁 Files reviewed ({len(archivos_rev)}):**")
                                    for a in archivos_rev:
                                        st.code(a, language=None)
                                if hallazgos:
                                    st.markdown(f"**🔍 Findings ({len(hallazgos)}):**")
                                    for i, h in enumerate(hallazgos, 1):
                                        st.markdown(f"{i}. {h}")
                                st.markdown("**📋 Plan:**")
                                st.write(p.get("plan", "No details"))

                                # Action buttons inside the plan
                                if t.status == "AWAITING_VALIDATION":
                                    st.info("⏳ PR submitted — ticket awaiting validation.")
                                else:
                                    btn_c1, btn_c2 = st.columns(2)
                                    with btn_c1:
                                        if st.button(f"🚀 Execute Plan V{v_num}", key=f"exec_plan_{v_num}", use_container_width=True, type="primary"):
                                            from agent_engine import ejecutar_plan_accion
                                            with st.spinner("🔧 Generating code on branch..."):
                                                res_exec = ejecutar_plan_accion.invoke({"ticket_id": t_id})
                                            if "Error" in res_exec:
                                                db.add(TicketThread(id=str(uuid.uuid4()), ticket_id=t_id, author="SRE-Agent", content=f"**Error executing plan:** {res_exec}"))
                                                db.commit()
                                            st.rerun()
                                    with btn_c2:
                                        if st.button(f"📬 Send PR V{v_num}", key=f"pr_plan_{v_num}", use_container_width=True):
                                            from agent_engine import crear_pr_ticket
                                            with st.spinner("📬 Creating Pull Request..."):
                                                res_pr = crear_pr_ticket.invoke({"ticket_id": t_id})
                                            if "Error" in res_pr:
                                                db.add(TicketThread(id=str(uuid.uuid4()), ticket_id=t_id, author="SRE-Agent", content=f"**Error creating PR:** {res_pr}"))
                                                db.commit()
                                            st.rerun()

                with c2:
                    st.markdown("#### 💬 History")
                    hilos = db.query(TicketThread).filter(TicketThread.ticket_id == t_id).order_by(TicketThread.timestamp.asc()).all()
                    if not hilos:
                        st.info("No activity yet.")
                    for h in hilos:
                        with st.chat_message("assistant" if h.author == "SRE-Agent" else "user"):
                            st.caption(f"{h.author} — {h.timestamp.strftime('%Y-%m-%d %H:%M')}")
                            st.write(h.content)
                    with st.form(key=f"form_thread_{t_id}", clear_on_submit=True):
                        new_txt = st.text_area("Add a comment...")
                        if st.form_submit_button("Submit"):
                            if new_txt:
                                db.add(TicketThread(id=str(uuid.uuid4()), ticket_id=t_id, author="SRE-Admin", content=new_txt))
                                db.commit()
                                st.rerun()

            else:
                st.session_state.selected_ticket = None
        else:
            cols = st.columns(5)
            statuses = [("Abierto", "OPEN"), ("IN_PROGRESS", "IN PROGRESS"), ("PENDING_NOTIF", "REVIEW"), ("AWAITING_VALIDATION", "VALIDATION"), ("RESOLVED", "RESOLVED")]
            for i, (status_id, label) in enumerate(statuses):
                with cols[i]:
                    st.markdown(f"### {label}")
                    tkts = db.query(Ticket).filter(Ticket.status == status_id).all()
                    for tk in tkts:
                        with st.container(border=True):
                            st.markdown(f"**{tk.id}**")
                            st.caption(f"👤 {tk.assigned_to}")
                            if st.button("📂 View Details", key=f"btn_{tk.id}", use_container_width=True):
                                st.session_state.selected_ticket = tk.id
                                st.rerun()
    finally:
        db.close()

# --- SECTION: KNOWLEDGE BASE ---
elif st.session_state.section == "Knowledge Base":
    st.header("📚 GitHub Knowledge Base")
    db = SessionLocal()
    try:
        repos = db.query(Repository).all()
        for r in repos:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.write(f"🔗 **{r.url}**")
                c1.caption(f"Updated: {r.last_updated}")
                if c2.button("🔄 Refresh", key=f"upd_{r.id}", use_container_width=True):
                    r.last_updated = datetime.utcnow()
                    db.commit()
                    st.toast("Refresh started...")
                    st.rerun()
        st.divider()
        st.markdown("#### Link new repository")
        url_input = st.text_input("GitHub URL")
        if st.button("📥 Sync New", type="primary"):
            if url_input:
                new_r = Repository(id=str(uuid.uuid4()), url=url_input, last_updated=datetime.utcnow())
                db.add(new_r)
                db.commit()
                st.success("Repository linked successfully!")
                st.rerun()
    finally:
        db.close()

    # --- QDRANT DIAGNOSTICS ---
    st.divider()
    st.markdown("#### 🔬 Vector Database Diagnostics (Qdrant)")
    if st.button("🩺 Inspect Qdrant", use_container_width=True):
        from agent_engine import diagnosticar_qdrant
        with st.spinner("Inspecting kb_sre collection..."):
            diag = diagnosticar_qdrant()

        if not diag["collection_exists"]:
            st.error("❌ Collection `kb_sre` does NOT exist in Qdrant.")
        elif diag["total_points"] == 0:
            st.warning("⚠️ Collection exists but is EMPTY (0 points). No code indexed.")
        else:
            st.success(f"✅ Active collection: **{diag['total_points']} vectors** indexed")

            archivos = diag.get("archivos", {})
            if archivos:
                st.markdown(f"**📁 Indexed files ({len(archivos)}):**")
                for nombre, count in sorted(archivos.items(), key=lambda x: -x[1]):
                    st.text(f"  {nombre} → {count} chunks")
            else:
                st.warning("No file names found in metadata.")

            keys = diag.get("metadata_keys", [])
            if keys:
                st.markdown(f"**🔑 Metadata keys:** `{', '.join(sorted(keys))}`")

            samples = diag.get("sample_payloads", [])
            if samples:
                with st.expander("🧪 Sample payloads (first 3 points)", expanded=False):
                    for i, s in enumerate(samples):
                        st.json(s)

        if "error" in diag:
            st.error(f"Error: {diag['error']}")
