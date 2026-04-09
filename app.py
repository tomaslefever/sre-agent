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
    qdrant_url = os.getenv("QDRANT_URL", "http://sre_qdrant:6333")
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
    ("system", "Eres un asistente de IA útil e inteligente. Tienes acceso a herramientas para buscar información y crear tickets. Si no sabes algo, usa tus herramientas."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Función para obtener la conexión a la BD por cada ID de sesión
def get_session_history(session_id: str):
    db_url = os.getenv("DATABASE_URL", "postgresql://agentx:supersecret@localhost:5432/chat_history")
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
st.title("🤖 Asistente AgentX")
st.markdown("Agente inteligente con memoria persistente (PostgreSQL).")

# Generar un ID de sesión único si es un usuario nuevo (o podrías pedirle un username)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Cargar el historial desde PostgreSQL para mostrarlo en la pantalla
history = get_session_history(st.session_state.session_id)

for msg in history.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Entrada de nuevo mensaje
if prompt_usuario := st.chat_input("Escribe tu mensaje aquí..."):
    with st.chat_message("user"):
        st.markdown(prompt_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            # Al invocar, el historial se lee de Postgres, se envía a OpenAI y se guarda de vuelta automáticamente
            respuesta = agent_with_chat_history.invoke(
                {"input": prompt_usuario},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            st.markdown(respuesta["output"])