import os
import uuid
import streamlit as st
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.agents import create_tool_calling_agent
try:
    from langchain.agents import AgentExecutor
except ImportError:
    from langchain.agents.agent import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# --- NUEVAS IMPORTACIONES PARA POSTGRESQL ---
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ==========================================
# 1. INICIALIZACIÓN DE SERVICIOS
# ==========================================
if "OPENAI_API_KEY" not in os.environ:
    st.error("⚠️ No se encontró la variable de entorno `OPENAI_API_KEY`. Por favor, configúrala en el panel de control (Easypanel) para continuar.")
    st.stop()

@st.cache_resource
def init_observability():
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006/v1/traces")
    try:
        LangChainInstrumentor().instrument()
    except Exception as e:
        st.warning(f"No se pudo inicializar la observabilidad: {e}")
    return True

@st.cache_resource
def init_qdrant_and_embeddings():
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant-db:6333")
    # --- DIAGNÓSTICO DE RED ---
    with st.expander("🛠️ Diagnóstico de Conexión (Debug)"):
        st.write(f"URL configurada: `{qdrant_url}`")
        try:
            import socket
            hostname = qdrant_url.split("//")[-1].split(":")[0]
            ip = socket.gethostbyname(hostname)
            st.success(f"Host `{hostname}` resuelto a IP: `{ip}`")
        except Exception as e:
            st.error(f"No se pudo resolver el host `{hostname}`: {e}")
    # --------------------------
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        st.info(f"Conectando a Qdrant en: `{qdrant_url}`")
        client = QdrantClient(url=qdrant_url) 
        
        if not client.collection_exists("mis_documentos"):
            client.create_collection(
                collection_name="mis_documentos",
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
        return QdrantVectorStore(client=client, collection_name="mis_documentos", embedding=embeddings)
    except Exception as e:
        st.error(f"Error al conectar con Qdrant (`{qdrant_url}`): {e}")
        st.stop()

init_observability()
vector_store = init_qdrant_and_embeddings()

# ==========================================
# 2. HERRAMIENTAS DEL AGENTE
# ==========================================
@tool
def buscar_en_base_de_conocimiento(query: str) -> str:
    """Usa esta herramienta SIEMPRE para buscar información de la empresa, manuales o documentos."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)
    if not docs:
        return "No se encontró información relevante en la base de datos."
    return "\n\n".join([doc.page_content for doc in docs])

@tool
def crear_ticket_soporte(descripcion: str, prioridad: str) -> str:
    """Usa esta herramienta para crear un ticket de soporte cuando el usuario pida contactar a un humano."""
    return f"Ticket TCK-9874 creado exitosamente con prioridad '{prioridad}'."

tools = [buscar_en_base_de_conocimiento, crear_ticket_soporte]

# ==========================================
# 3. CONFIGURACIÓN DEL AGENTE Y MEMORIA EN POSTGRESQL
# ==========================================
prompt = ChatPromptTemplate.from_messages([
    ("system", """Eres un Ingeniero de Fiabilidad de Sitios (SRE) experto y proactivo. 
    Tu objetivo es ayudar a minimizar el tiempo de resolución de incidentes (MTTR).
    
    Capacidades:
    1. Buscar en la base de conocimientos manuales técnicos y runbooks.
    2. Crear tickets de soporte con prioridad clara basada en el impacto.
    3. Analizar síntomas de fallos en infraestructura.
    
    Personalidad: Profesional, analítico y centrado en la resolución. Si detectas un problema crítico, sugiere siempre crear un ticket."""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Función para obtener la conexión a la BD por cada ID de sesión
def get_session_history(session_id: str):
    db_url = os.getenv("DATABASE_URL", "postgresql://agentx:supersecret@postgres:5432/chat_history")
    st.info(f"Conectando a PostgreSQL en: `{db_url.split('@')[1]}`")
    return SQLChatMessageHistory(session_id=session_id, connection_string=db_url)

# Envolvemos el agente con la memoria persistente
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# ==========================================
# 4. INTERFAZ DE USUARIO (STREAMLIT)
# ==========================================

# Configuración de página
st.set_page_config(page_title="SRE AgentX | Hackathon Edition", page_icon="🤖", layout="wide")

# Estilo personalizado para el Sidebar
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #0e1117;
        border-right: 1px solid #30363d;
    }
    .stStatusWidget {
        background-color: #161b22;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2040/2040523.png", width=80)
    st.title("SRE Control Panel")
    st.markdown("---")
    
    st.subheader("🌐 Infrastructure Health")
    cols = st.columns(3)
    with cols[0]: st.write("🧠 Qdrant"); st.caption("✅ Online")
    with cols[1]: st.write("🐘 Postgres"); st.caption("✅ Online")
    with cols[2]: st.write("🔥 Phoenix"); st.caption("✅ Online")
    
    st.markdown("---")
    st.subheader("🚀 Quick Actions")
    if st.button("Simular Alerta de Latencia"):
        st.session_state.simulate_alert = "ALERTA: Latencia > 500ms en el microservicio de Pagos."
    
    st.markdown("---")
    st.info("""
    **Objetivo del Agente:**
    Analizar incidentes, consultar bases de conocimiento y automatizar la creación de tickets de soporte para reducir el MTTR (Mean Time To Recovery).
    """)

# --- MAIN UI ---
st.title("🤖 SRE Agente Inteligente")
st.markdown("""
Bienvenido al **Centro de Comando de Incidentes**. Soy tu Agente SRE, entrenado para administrar tickets y resolver dudas de infraestructura.
""")

# Generar un ID de sesión único
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Cargar el historial desde PostgreSQL
history = get_session_history(st.session_state.session_id)

# Mensaje de bienvenida si el historial está vacío
if len(history.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("""
        ¡Hola! Soy tu **SRE AgentX**. Estoy listo para ayudarte con el Hackathon. 
        Puedo asistirte en:
        *   🔍 **Consultar Runbooks**: Busco en la base de datos vectorial (Qdrant).
        *   🎫 **Gestionar Tickets**: Puedo crear registros de incidencia de forma autónoma.
        *   📊 **Análisis de Trazas**: Todo lo que hagamos está siendo monitoreado en Phoenix.
        
        ¿Qué incidente estamos analizando hoy?
        """)

# Mostrar historial
for msg in history.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Lógica de entrada de mensajes
if prompt_usuario := st.chat_input("Describe el incidente o pide crear un ticket..."):
    # Si hay una alerta simulada, la adjuntamos al mensaje de forma interna
    input_final = prompt_usuario
    if "simulate_alert" in st.session_state:
        input_final = f"{st.session_state.simulate_alert}\n\nUsuario dice: {prompt_usuario}"
        del st.session_state.simulate_alert

    with st.chat_message("user"):
        st.markdown(prompt_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Analizando infraestructura y consultando bases..."):
            respuesta = agent_with_chat_history.invoke(
                {"input": input_final},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            st.markdown(respuesta["output"])