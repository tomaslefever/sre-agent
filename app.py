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
# 1.5 INICIALIZACIÓN DE VECTOR STORE Y RAG
# ==========================================
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
        pass # Handle if Qdrant throws error checking collection
    return QdrantVectorStore(client=client, collection_name="kb_sre", embedding=embeddings)

vector_store = get_vector_store()

# ==========================================
# 2. HERRAMIENTAS DEL AGENTE
# ==========================================
@tool
def buscar_conocimiento(query: str) -> str:
    """Importante: Usa esto para buscar cualquier error en los archivos '.log' subidos, síntomas o manuales de infraestructura."""
    docs = vector_store.as_retriever(search_kwargs={"k": 5}).invoke(query)
    resultados = []
    for d in docs:
        origen = d.metadata.get("source", "kb_general")
        resultados.append(f"[{origen}]: {d.page_content}")
    return "\n\n---\n\n".join(resultados) if resultados else "No se encontró coincidencia."

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
    if "last_upload" in st.session_state and st.session_state.last_upload:
        att = Attachment(id=str(uuid.uuid4()), ticket_id=t_id, 
                         filename=st.session_state.last_upload["name"],
                         file_type=st.session_state.last_upload["type"])
        db.add(att)
        st.session_state.last_upload = None

    db.commit()
    db.close()
    return f"✅ Ticket {t_id} creado y asignado a {asignado}. El incidente ha sido registrado."

@tool
def asignar_ticket(ticket_id: str, tecnico: str) -> str:
    """Reasigna un ticket a otro técnico."""
    db = SessionLocal()
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if ticket:
        ticket.assigned_to = tecnico
        db.commit()
        db.close()
        return f"✅ Ticket {ticket_id} reasignado a {tecnico}."
    db.close()
    return f"❌ Ticket {ticket_id} no encontrado."

@tool
def notificar_soporte(ticket_id: str, mensaje: str) -> str:
    """Envía una notificación interna al equipo de soporte sobre un ticket."""
    # Mock de envío de notificación
    return f"✅ Notificación de soporte enviada para {ticket_id}: '{mensaje}'."

@tool
def notificar_usuario(ticket_id: str, mensaje: str) -> str:
    """Envía un correo o alerta al usuario/autor informando el estado de su ticket."""
    # Mock de envío a usuario
    return f"✅ Usuario notificado sobre {ticket_id}: '{mensaje}'."

tools = [buscar_conocimiento, crear_ticket_sre, asignar_ticket, notificar_soporte, notificar_usuario]

# ==========================================
# 3. CEREBRO DEL AGENTE
# ==========================================
# Permitir que el modelo se configure por variable de entorno para usar modelos más potentes/nuevos 
modelo_llm = os.getenv("OPENAI_MODEL", "gpt-4o")
llm = ChatOpenAI(model=modelo_llm, temperature=0)
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
# 4. INTERFAZ: SIDEBAR Y NAVEGACIÓN
# ==========================================
with st.sidebar:
    st.header("SRE Control Center")
    
    # Navegación mediante botones en vez de select/radio
    if "seccion" not in st.session_state:
        st.session_state.seccion = "Centro de Incidentes"
        
    if st.button("🤖 Centro de Incidentes", use_container_width=True):
        st.session_state.seccion = "Centro de Incidentes"
    if st.button("📊 Tablero de Tickets", use_container_width=True):
        st.session_state.seccion = "Tablero de Tickets"
    
    st.markdown("---")
    
    st.session_state.debug_mode = st.checkbox("🛠️ Activar Debug Mode", value=st.session_state.get("debug_mode", False))
    
    st.markdown("---")
    
    # Listado de Sesiones como botones y sin la palabra "Sesiones"
    if st.button("+ Nueva Sesión", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())
        st.session_state.session_list.append(new_id)
        st.session_state.session_id = new_id
        st.rerun()
    
    for s_id in reversed(st.session_state.session_list):
        label = f"➡️ Sesión activa" if s_id == st.session_state.session_id else f"Historial {s_id[:6]}"
        if st.button(label, key=f"btn_{s_id}", use_container_width=True):
            st.session_state.session_id = s_id
            st.rerun()

# ==========================================
# 5. VISTA PRINCIPAL
# ==========================================
st.title("AgentX: SRE Intelligence Platform")

if st.session_state.seccion == "Centro de Incidentes":
    # Mostrar historial del chat
    h = get_chat_history(st.session_state.session_id)
    for m in h.messages:
        role = "user" if m.type == "human" else "assistant"
        with st.chat_message(role): 
            content = m.content
            # Extraer y ocultar la parte del sistema en los logs cargados
            if "[SISTEMA:" in content:
                partes = content.split("[SISTEMA:", 1)
                visible = partes[0].strip()
                hidden = "[SISTEMA:" + partes[1]
                if visible:
                    st.markdown(visible)
                if st.session_state.debug_mode:
                    with st.expander("🔎 Info de Sistema (Debug)"):
                        st.caption(hidden)
            else:
                st.markdown(content)

    # El dropzone colapsado flotando en el fondo pegado al input
    up_file = None
    try:
        # st.container(bottom=True) hace que el contenedor se pegue abajo junto al chat input (Streamlit > 1.30)
        with st.container(bottom=True):
            st.markdown("<style>div[data-testid='stFileUploader'] {margin-bottom: -15px;}</style>", unsafe_allow_html=True)
            up_file = st.file_uploader("Evidencia", type=["txt", "log", "png", "jpg", "jpeg", "mp4"], label_visibility="collapsed")
            if up_file:
                st.session_state.last_upload = {"name": up_file.name, "type": up_file.type}
    except TypeError:
        # Fallback si Streamlit es una versión más antigua
        with st.container():
            st.markdown("<style>div[data-testid='stFileUploader'] {margin-bottom: -15px;}</style>", unsafe_allow_html=True)
            up_file = st.file_uploader("Evidencia", type=["txt", "log", "png", "jpg", "jpeg", "mp4"], label_visibility="collapsed")
            if up_file:
                st.session_state.last_upload = {"name": up_file.name, "type": up_file.type}
            
    # Input y procesamiento del chat
    if u_input := st.chat_input("Diagnostica un fallo o solicita un ticket..."):
        ctx = ""
        sys_msg = ""
        if up_file:
            if up_file.name.endswith((".log", ".txt")):
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                from langchain_core.documents import Document
                
                file_content = up_file.read().decode(errors='replace')
                # 1. Separamos el log gigante en fracciones navegables (Chunks)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
                chunks = text_splitter.split_text(file_content)
                docs = [Document(page_content=c, metadata={"source": up_file.name}) for c in chunks]
                
                # 2. Ingesta a Qdrant (Base de datos vectorial)
                with st.spinner(f"Indexando archivo gigante de {len(chunks)} fragmentos en Qdrant..."):
                    vector_store.add_documents(docs)
                
                # 3. Guardamos el mensaje de sistema a agregar al prompt
                sys_msg = f"\n\n[SISTEMA: El usuario ha adjuntado el archivo log '{up_file.name}'. No está en este prompt. En su lugar, ha sido indexado en tu base vectorial (Qdrant). Usa de forma obligatoria tu herramienta 'buscar_conocimiento' para leer sus partes y responder a la pregunta del usuario.]"
            else:
                sys_msg = f"\n\n[SISTEMA: MULTIMEDIA ADJUNTA: {up_file.name}]"
        
        full_input = u_input + sys_msg
        
        # Ocultar la parte de sistema en la UI en vivo
        with st.chat_message("user"): 
            st.markdown(u_input)
            if up_file:
                if st.session_state.debug_mode:
                    with st.expander("🔎 Info de Sistema (Debug)"):
                        st.caption(sys_msg)
                else:
                    st.caption(f"📎 *Evidencia vinculada: {up_file.name}*")
        
        with st.chat_message("assistant"):
            with st.spinner("Analizando evidencia e infraestructura..."):
                res = agent_with_memory.invoke(
                    {"input": full_input},
                    config={"configurable": {"session_id": st.session_state.session_id}}
                )
                st.markdown(res["output"])

elif st.session_state.seccion == "Tablero de Tickets":
    st.header("Sistema de Gestión de Tickets")
    db = SessionLocal()
    
    tickets_df = pd.read_sql(db.query(Ticket).statement, engine)
    if not tickets_df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Tickets Abiertos", len(tickets_df[tickets_df['status'] == 'Abierto']))
        c2.metric("Tickets Totales", len(tickets_df))
        c3.metric("Técnicos Activos", len(tickets_df['assigned_to'].unique()))
        st.markdown("---")
        
    all_t = db.query(Ticket).all()
    
    if all_t:
        for t in all_t:
            with st.expander(f"{t.id} - {t.report[:80]}..."):
                cols = st.columns(3)
                cols[0].metric("Autor", t.author)
                cols[1].metric("Asignado a", t.assigned_to)
                cols[2].metric("Estado", t.status)
                
                st.write("**Reporte de Incidente:**")
                st.info(t.report)
                
                # Mostrar adjuntos si los hay
                atts = db.query(Attachment).filter_by(ticket_id=t.id).all()
                if atts:
                    st.write("**Adjuntos guardados en el ticket:**")
                    for a in atts:
                        st.markdown(f"📎 `{a.filename}` ({a.file_type})")
    else:
        st.info("El tablero está vacío. Puedes pedir al agente que registre un incidente.")
    db.close()