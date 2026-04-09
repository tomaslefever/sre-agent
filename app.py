import os
import uuid
import base64
import streamlit as st
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# --- PERSISTENCIA ---
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ==========================================
# 1. CONFIGURACIÓN Y ESTILO (TAILWIND)
# ==========================================
st.set_page_config(page_title="SRE AgentX Platform", page_icon="🤖", layout="wide")

# Inyectamos Tailwind para futuras expansiones de UI personalizada
st.markdown('<script src="https://cdn.tailwindcss.com"></script>', unsafe_allow_html=True)

if "OPENAI_API_KEY" not in os.environ:
    st.error("⚠️ Falta OPENAI_API_KEY en las variables de entorno.")
    st.stop()

# Manejo de IDs de Sesión
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "session_list" not in st.session_state:
    st.session_state.session_list = [st.session_state.session_id]

# ==========================================
# 2. INICIALIZACIÓN DE SERVICIOS
# ==========================================
@st.cache_resource
def init_qdrant():
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    client = QdrantClient(url=qdrant_url) 
    if not client.collection_exists("kb_sre"):
        client.create_collection(
            collection_name="kb_sre",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    return QdrantVectorStore(client=client, collection_name="kb_sre", embedding=embeddings)

vector_store = init_qdrant()

# ==========================================
# 3. HERRAMIENTAS Y CEREBRO DEL AGENTE
# ==========================================
@tool
def diagnosticar_incidente(query: str) -> str:
    """Busca en la base de conocimiento interna soluciones a errores conocidos."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs]) if docs else "No se encontró historial previo de este error."

@tool
def registrar_ticket(descripcion: str, prioridad: str) -> str:
    """Crea un ticket oficial con ID único para seguimiento humano."""
    return f"Ticket generado: [#{uuid.uuid4().hex[:6].upper()}] - Prioridad: {prioridad}. El equipo SRE ha sido alertado."

tools = [diagnosticar_incidente, registrar_ticket]

# El prompt ahora acepta instrucciones multimodales (textuales por ahora para compatibilidad)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres AgentX, un Ingeniero SRE avanzado. Analizas texto, logs y descripciones de imágenes. Tu misión es resolver problemas y reducir el ruido en las alertas."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

llm = ChatOpenAI(model="gpt-4o", temperature=0) # GPT-4o es nativamente multimodal
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def get_history(session_id: str):
    db_url = os.getenv("DATABASE_URL", "postgresql://agentx:supersecret@postgres:5432/chat_history")
    return SQLChatMessageHistory(session_id=session_id, connection_string=db_url)

agent_fluent = RunnableWithMessageHistory(
    agent_executor, get_history,
    input_messages_key="input", history_messages_key="chat_history"
)

# ==========================================
# 4. INTERFAZ (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("SRE Control Panel")
    st.markdown("---")
    
    st.subheader("Sesiones Activas")
    if st.button("+ Nueva Sesión", type="primary", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.session_list.append(new_id)
        st.session_state.session_id = new_id
        st.rerun()
    
    selected_session = st.selectbox("Historial de Incidencias", st.session_state.session_list, 
                                   index=st.session_state.session_list.index(st.session_state.session_id))
    if selected_session != st.session_state.session_id:
        st.session_state.session_id = selected_session
        st.rerun()

    st.markdown("---")
    st.subheader("Evidencia Multimodal")
    uploaded_file = st.file_uploader("Subir Logs o Imágenes", type=["txt", "log", "png", "jpg", "jpeg"])
    if st.checkbox("Modo Debug (Red/Logs)"):
        st.caption(f"ID Sesión: `{st.session_state.session_id}`")
        st.caption(f"Qdrant: `{os.getenv('QDRANT_URL', 'http://qdrant-db:6333')}`")

# ==========================================
# 5. ÁREA DE CHAT PRINCIPAL
# ==========================================
st.title("AgentX: SRE Intelligence")

# Mostrar historial persistente
history_store = get_history(st.session_state.session_id)
for msg in history_store.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Input del usuario
if user_input := st.chat_input("Escribe el problema o pregunta sobre los logs..."):
    
    contexto_archivo = ""
    # Procesamiento de archivos (Multimodal)
    if uploaded_file:
        if uploaded_file.type.startswith("image/"):
            # Para imágenes, la describimos contextualmente mientras LangChain mejora su soporte nativo directo en agentes
            contexto_archivo = f"\n[ANALIZANDO IMAGEN ADJUNTA: {uploaded_file.name}]"
        elif uploaded_file.name.endswith((".log", ".txt")):
            log_txt = uploaded_file.read().decode()
            contexto_archivo = f"\n\n--- LOG ADJUNTO ({uploaded_file.name}) ---\n{log_txt}\n------------------\n"

    with st.chat_message("user"):
        st.markdown(user_input + contexto_archivo)

    with st.chat_message("assistant"):
        with st.spinner("Procesando evidencia técnica..."):
            full_query = user_input + contexto_archivo
            respuesta = agent_fluent.invoke(
                {"input": full_query},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            st.markdown(respuesta["output"])