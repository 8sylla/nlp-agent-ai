# 🤖 GraphRAG AI Support Agent
### Enterprise-Grade Hybrid Conversational AI (Neuro-Symbolic Architecture)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-13+-black.svg?logo=next.js&logoColor=white)](https://nextjs.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-Graph_DB-008CC1.svg?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![LLM Agnostic](https://img.shields.io/badge/AI-Gemini_%7C_Groq-purple.svg?logo=openai&logoColor=white)](https://python.langchain.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

> **A next-generation customer support agent capable of reasoning, maintaining context, and handling complex relational queries by orchestrating Knowledge Graphs (GraphRAG), Vector Search (VectorRAG), and Transactional APIs.**

---

## 📖 Executive Summary

Traditional chatbots fail when faced with complex, relational queries (e.g., *"Is this charger compatible with my phone?"*) or multi-turn context. They suffer from hallucinations and "context blindness."

This project solves this by implementing a **Hybrid Neuro-Symbolic Architecture**:

1.  **Transactional Engine (Deterministic):** Handles order tracking instantly via Regex/NLU and Mock ERP APIs.
2.  **Reasoning Engine (GraphRAG):** Uses **Neo4j** and **LLMs (Gemini/Llama 3)** to traverse a Knowledge Graph for logical answers (compatibility, hierarchy, warranty).
3.  **Semantic Engine (VectorRAG):** Uses **PostgreSQL/pgvector** as a fallback for unstructured documentation (FAQ, Policies).

It features **Contextual Memory** (Redis), **Multilingual Support** (French/Arabic), and **Sentiment Analysis** for empathetic responses.

---

## Architecture

The system is built on a Microservices architecture, fully containerized with Docker.

![Architecture](/docs/archi-fulla.png)

### Key Technical Features
*   **LLM Agnostic:** Switch between **Google Gemini 2.5 Flash** (Cost-effective) and **Groq Llama 3.3** (Low Latency) via environment variables.
*   **GraphRAG Ingestion (ETL):** An automated pipeline using LLMs to extract Entities and Relationships from raw text into Neo4j.
*   **Contextual Rephrasing:** Uses LLMs to rewrite follow-up questions (e.g., *"And its price?"* becomes *"What is the price of iPhone 15?"*) before querying databases.
*   **Real-Time Dashboard:** Live monitoring of conversations, sentiment scores, and AI reasoning steps.

---

## 🛠️ Tech Stack

| Component | Technology | Role |
|-----------|------------|------|
| **Backend** | **Python 3.11 / FastAPI** | Asynchronous API & WebSocket Orchestrator. |
| **Frontend** | **Next.js 13+ / Tailwind** | Modern, responsive Chat UI & Admin Dashboard. |
| **AI Core** | **LangChain 0.3+** | Orchestration framework. |
| **LLM Provider** | **Gemini 1.5** or **Groq** | Configurable inference engine. |
| **Graph DB** | **Neo4j 5.x** | Storing structured knowledge (Products, Relations). |
| **Vector DB** | **PostgreSQL (pgvector)** | Storing semantic embeddings. |
| **Memory** | **Redis** | Storing conversation history (Short-term memory). |

---

## 🚀 Getting Started

### 1. Prerequisites
*   **Docker** & **Docker Compose** installed.
*   **Make** (Optional, for easy commands).
*   API Keys: **Google Gemini** (Free tier) OR **Groq** (Free beta).

### 2. Installation


Clone the repository:
```bash
git clone https://github.com/8sylla/ai-support-agent.git
cd ai-support-agent
```

### 3. Environment Configuration

Create a `.env` file in the root directory. Choose your LLM provider.

```ini
# --- DATABASE CONFIG ---
POSTGRES_USER=admin
POSTGRES_PASSWORD=adminpassword
POSTGRES_DB=agent_db

# --- NEO4J CONFIG ---
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password1234

# --- AI PROVIDER SELECTION ---
# Options: 'google' or 'groq'
LLM_PROVIDER=google

# Google Config
GOOGLE_API_KEY=AIzaSyDxxxxxxxxxxxxxxxxxxxxxxxx
GOOGLE_MODEL=gemini-1.5-flash

# Groq Config (Optional)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_MODEL=llama-3.3-70b-versatile
```

### 4. Build & Run (The Easy Way)

We use a `Makefile` to simplify Docker management.

```bash
# Build and Start the stack
make install
make start
```

*The application will be available at:*
*    **Frontend:** http://localhost:3000
*    **Backend:** http://localhost:8000/docs
*    **Admin:** http://localhost:3000/admin
*   🕸️ **Neo4j:** http://localhost:7474

### 5. Data Ingestion (Crucial Step)

The databases are empty initially. You must run the ETL pipelines to populate the Knowledge Graph and Vector Index.

```bash
# Runs both Graph Ingestion (Neo4j) and Vector Ingestion (Postgres)
make ingest-all
```

---

## 🖥️ Usage Scenarios

| Engine | Trigger Example | Expected Outcome |
|--------|-----------------|------------------|
| **Transactional** | *"Where is order CMD-123?"* | Returns real-time status from Mock ERP. |
| **GraphRAG** | *"Who manufactures the iPhone 15?"* | Traverses graph `(iPhone 15)-[MANUFACTURED_BY]->(Apple)`. |
| **Reasoning** | *"Is the USB-C cable compatible with iPhone 15?"* | Checks compatibility path in Neo4j. |
| **Memory** | *"What is its warranty?"* (after iPhone question) | Rewrites query to *"What is the warranty of iPhone 15?"*. |
| **VectorRAG** | *"Do you deliver to Morocco?"* | Finds semantic match in FAQ documentation. |
| **Multilingual** | *"من يصنع الآيفون؟"* | Detects Arabic, queries Knowledge Base, answers in Arabic. |

---

## Developer Commands (Makefile)

| Command | Description |
|---------|-------------|
| `make start` | Starts the full stack in detached mode. |
| `make stop` | Stops all containers. |
| `make logs` | Shows realtime logs from the Backend API. |
| `make ingest-all` | Runs all ETL scripts (Graph + Vector + Arabic). |
| `make train-nlu` | Re-trains the Spacy NLU model and restarts API. |
| `make db-update` | Updates PostgreSQL schema (e.g. adds feedback column). |
| `make test` | Runs the Unit Test Suite (Pytest). |

---

## 📂 Project Structure

```bash
ai-support-agent/
├── backend/                 # FastAPI Application
│   ├── app/
│   │   ├── core/            # Intelligence Engines
│   │   │   ├── orchestrator.py  # MAIN LOGIC (Hybrid Router)
│   │   │   ├── graph_engine.py  # Neo4j + LLM Logic
│   │   │   ├── llm_loader.py    # Provider Factory (Google/Groq)
│   │   │   └── ...
│   ├── ingest_graph.py      # ETL Pipeline (Text -> Graph)
│   └── requirements-core.txt # Stable dependencies
├── frontend-next/           # Next.js Application
│   ├── app/                 # Pages (Chat & Admin Dashboard)
│   └── components/          # UI Components (OrderCard, Feedback)
├── docker-compose.yml       # Infrastructure orchestration
├── Makefile                 # Automation shortcuts
└── README.md                # You are here
```
