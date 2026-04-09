import os
import uuid
import base64
import pandas as pd
import streamlit as st
from datetime import datetime
from sqlalchemy import create_engine, Column, String, DateTime, Text, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# --- PERSISTENCIA CHAT ---
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ==========================================
# 0. CONFIGURACIÓN DE DB (TICKETS)
# ==========================================
Base = declarative_base()
DB_URL = os.getenv("DATABASE_URL", "postgresql://agentx:supersecret@postgres:5432/chat_history")
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(String, primary_key=True)
    report = Column(Text)
    author = Column(String)
    assigned_to = Column(String)
    status = Column(String, default="Abierto")
    created_at = Column(DateTime, default=datetime.utcnow)

class Attachment(Base):
    __tablename__ = "attachments"
    id = Column(String, primary_key=True)
    ticket_id = Column(String, ForeignKey("tickets.id"))
    filename = Column(String)
    file_type = Column(String)

Base.metadata.create_all(engine)

# Lista de técnicos mock
TECNICOS = ["Alex SRE", "Sonia DevOps", "Carlos Cloud", "Marta Security"]

# ==========================================
# 1. CONFIGURACIÓN STREAMLIT
# ==========================================
st.set_page_config(page_title="AgentX: SRE & Ticketing", page_icon="🎫", layout="wide")
st.markdown('<script src="https://cdn.tailwindcss.com"></script>', unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "session_list" not in st.session_state:
    st.session_state.session_list = [st.session_state.session_id]

# ==========================================
# 2. HERRAMIENTAS DEL AGENTE
# ==========================================
@tool
def buscar_conocimiento(query: str) -> str:
    """Busca soluciones técnicas en la base de datos vectorial."""
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
    client = QdrantClient(url=qdrant_url)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = QdrantVectorStore(client=client, collection_name="kb_sre", embedding=embeddings)
    docs = vector_store.as_retriever().invoke(query)
    return "\n\n".join([d.page_content for d in docs]) if docs else "No hay info."

@tool
def crear_ticket_sre(reporte: str, autor: str, asignado: str = None) -> str:
    """Crea un ticket en el sistema. Si no se especifica asignado, elige uno de: Alex, Sonia, Carlos, Marta."""
    if not asignado:
        import random
        asignado = random.choice(TECNICOS)
    
    t_id = f"TCK-{uuid.uuid4().hex[:6].upper()}"
    db = SessionLocal()
    new_ticket = Ticket(id=t_id, report=reporte, author=autor, assigned_to=asignado)
    db.add(new_ticket)
    
    # Vincular archivo si hay uno en el estado temporal
    if "last_upload" in st.session_state:
        att = Attachment(id=str(uuid.uuid4()), ticket_id=t_id, 
                         filename=st.session_state.last_upload["name"],
                         file_type=st.session_state.last_upload["type"])
        db.add(att)
        del st.session_state.last_upload

    db.commit()
    db.close()
    return f"✅ Ticket {t_id} creado y asignado a {asignado}. El incidente ha sido registrado."

tools = [buscar_conocimiento, crear_ticket_sre]

# ==========================================
# 3. CEREBRO DEL AGENTE
# ==========================================
llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", f"Eres un Agente SRE experto. Administras incidencias y tickets. Técnicos disponibles: {', '.join(TECNICOS)}."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent_exec = AgentExecutor(agent=create_tool_calling_agent(llm, tools, prompt), tools=tools, verbose=True)

def get_chat_history(id_sesion: str):
    return SQLChatMessageHistory(session_id=id_sesion, connection_string=DB_URL)

agent_with_memory = RunnableWithMessageHistory(
    agent_exec, get_chat_history,
    input_messages_key="input", history_messages_key="chat_history"
)

# ==========================================
# 4. INTERFAZ: SIDEBAR Y DASHBOARD
# ==========================================
with st.sidebar:
    st.header("SRE Control Center")
    tab_chat, tab_tickets = st.tabs(["💬 Chat", "🎫 Tickets"])
    
    with tab_chat:
        st.subheader("Sesiones")
        if st.button("+ Nueva Sesión"):
            new_id = str(uuid.uuid4()); st.session_state.session_list.append(new_id)
            st.session_state.session_id = new_id; st.rerun()
        
        st.session_state.session_id = st.selectbox("Historial", st.session_state.session_list, 
                                                  index=st.session_state.session_list.index(st.session_state.session_id))
        st.markdown("---")
        st.subheader("Adjuntos")
        up_file = st.file_uploader("Evidencia (Logs/Img)", type=["txt", "log", "png", "jpg"])
        if up_file:
            st.session_state.last_upload = {"name": up_file.name, "type": up_file.type}

    with tab_tickets:
        st.subheader("Estado del Sistema")
        db = SessionLocal()
        tickets_df = pd.read_sql(db.query(Ticket).statement, engine)
        db.close()
        if not tickets_df.empty:
            st.dataframe(tickets_df[["id", "status", "assigned_to"]], hide_index=True)
        else:
            st.write("No hay tickets activos.")

# ==========================================
# 5. VISTA PRINCIPAL
# ==========================================
st.title("AgentX: SRE Intelligence Platform")

# Tabs principales
t1, t2 = st.tabs(["🤖 Centro de Incidentes", "📊 Tablero de Tickets"])

with t1:
    h = get_chat_history(st.session_state.session_id)
    for m in h.messages:
        role = "user" if m.type == "human" else "assistant"
        with st.chat_message(role): st.markdown(m.content)

    if u_input := st.chat_input("Diagnostica un fallo o solicita un ticket..."):
        ctx = ""
        if up_file:
            if up_file.name.endswith((".log", ".txt")):
                ctx = f"\n\n[ARCHIVO: {up_file.name}]\n{up_file.read().decode()}"
            else:
                ctx = f"\n\n[IMAGEN ADJUNTA: {up_file.name}]"
        
        with st.chat_message("user"): st.markdown(u_input + (ctx if len(ctx) < 500 else "\n[Log extenso adjunto]"))
        
        with st.chat_message("assistant"):
            with st.spinner("Analizando y procesando incidente..."):
                res = agent_with_memory.invoke(
                    {"input": u_input + ctx},
                    config={"configurable": {"session_id": st.session_state.session_id}}
                )
                st.markdown(res["output"])

with t2:
    st.header("🎫 Sistema de Gestión de Tickets (SRE)")
    db = SessionLocal()
    all_t = db.query(Ticket).all()
    
    if all_t:
        for t in all_t:
            with st.expander(f"{t.id} - {t.report[:50]}..."):
                c1, c2, c3 = st.columns(3)
                c1.metric("Autor", t.author)
                c2.metric("Asignado", t.assigned_to)
                c3.metric("Estado", t.status)
                st.write("**Reporte Completo:**")
                st.write(t.report)
                
                # Mostrar adjuntos
                atts = db.query(Attachment).filter_by(ticket_id=t.id).all()
                if atts:
                    st.write("**Adjuntos:**")
                    for a in atts:
                        st.caption(f"📎 {a.filename} ({a.file_type})")
    else:
        st.info("El tablero está vacío. Pide al agente que cree un ticket para empezar.")
    db.close()