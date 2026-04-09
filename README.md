# 🤖 SRE Agent - AgentX

*[Leer en Español abajo / Read in Spanish below](#español)*

An intelligent agent designed for SRE tasks and technical support, featuring persistent memory and advanced observability.

## 🚀 Features

- **Intelligent Agent:** Built with LangChain and OpenAI GPT-4o-mini.
- **Persistent Memory:** Chat history is automatically saved in **PostgreSQL**.
- **Knowledge Base:** Retrieval of relevant information using **Qdrant Vector DB**.
- **Observability:** Full trace and performance monitoring with **Arize Phoenix**.
- **Containerization:** Easy deployment with **Docker** and **Docker Compose**.

## 🛠️ Technologies

- ![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
- ![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
- ![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
- ![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-black)
- ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue?logo=postgresql)
- ![Docker](https://img.shields.io/badge/Docker-Containers-blue?logo=docker)

## 📖 Documentation

- [**Quick Start Guide** (QUICKGUIDE.md)](./QUICKGUIDE.md)
- [**Agent Usage** (AGENTS_USE.md)](./AGENTS_USE.md)
- [**Scaling Guide** (SCALING.md)](./SCALING.md)

## 🏁 Quick Start

1. Create your `.env` file based on `env.example`.
2. Run `docker compose up --build`.
3. Open [http://localhost:8501](http://localhost:8501).

---
---

<a name="español"></a>
# 🤖 SRE Agent - AgentX (Español)

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
Desarrollado por / Developed by [tomaslefever](https://github.com/tomaslefever)