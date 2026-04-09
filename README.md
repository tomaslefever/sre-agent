# 🤖 SRE Agent - AgentX

Un agente inteligente diseñado para tareas de SRE y soporte técnico, con memoria persistente y observabilidad avanzada.

## 🚀 Características

- **Agente Inteligente:** Basado en LangChain y OpenAI GPT-4o-mini.
- **Memoria Persistente:** Historial de chat guardado automáticamente en **PostgreSQL**.
- **Base de Conocimientos:** Recuperación de información relevante usando **Qdrant Vector DB**.
- **Observabilidad:** Seguimiento completo de trazas y rendimiento con **Arize Phoenix**.
- **Contenerización:** Despliegue sencillo con **Docker** y **Docker Compose**.

## 🛠️ Tecnologías

- ![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
- ![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
- ![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
- ![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-black)
- ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue?logo=postgresql)
- ![Docker](https://img.shields.io/badge/Docker-Containers-blue?logo=docker)

## 📖 Documentación

- [**Guía de Inicio Rápido** (QUICKGUIDE.md)](./QUICKGUIDE.md)
- [**Uso de Agentes** (AGENTS_USE.md)](./AGENTS_USE.md)
- [**Escalamiento** (SCALING.md)](./SCALING.md)

## 🏁 Inicio Rápido

1. Crea tu archivo `.env` basado en `env.example`.
2. Ejecuta `docker compose up --build`.
3. Abre [http://localhost:8501](http://localhost:8501).

---
Desarrollado por [tomaslefever](https://github.com/tomaslefever)