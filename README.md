# 🧠 Corporate Institutional Memory System

 A multi-agent AI system that captures, connects, and preserves an organisation's collective knowledge — decisions, relationships, lessons learned, and expertise before it walks out the door.

Built as a deep-dive learning project into **multi-agent orchestration**, **LangGraph state machines**, **hybrid RAG (vector + graph)**, and **production-grade agentic system design**.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📌 The Problem

Every organisation loses years of institutional knowledge silently:

- **Why** was a vendor dropped? Nobody remembers.
- **Who** decided this policy, and what alternatives were rejected?
- A key employee resigns — and years of undocumented process knowledge leaves with them.
- Post-mortems are written, filed away, and never referenced again.

This system solves that by building a **living, queryable second brain** for the organisation — powered by a coordinated team of AI agents rather than a single monolithic chatbot.

----------------------------------------------------------------------------------------------------------------------------

## 🏗️ Architecture Overview

```
                          ┌─────────────────────────┐
                          │   Master Orchestrator    │
                          │   (top-level router)     │
                          └────────────┬─────────────┘
                                       │
        ┌──────────────────┬──────────┴──────────┬──────────────────┐
        ▼                  ▼                     ▼                  ▼
┌───────────────┐  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Retrieval     │  │  Capture       │   │  Audit         │   │  Health        │
│  Orchestrator  │  │  Orchestrator  │   │  Orchestrator  │   │  Check         │
└───────┬────────┘  └───────┬────────┘   └───────┬────────┘  └────────────────┘
        │                   │                     │
   ┌────┴────┐         ┌────┴─────┐          ┌────┴─────┐
   ▼         ▼         ▼          ▼          ▼          ▼
Router   Decision   Meeting   PostMortem  GapDetector  StalenessAgent
Agent    People     Tribal    Agent       SPFAgent
         Policy     Knowledge
         Project    Agent
         Competitive
```

### Three Sub-Orchestrators, One Brain

| Orchestrator | Responsibility | Agents Coordinated |
|---|---|---|
| **Retrieval Orchestrator** | Answers user questions | Router, Decision, People, Policy, Project, Competitive |
| **Capture Orchestrator** | Ingests new knowledge | Meeting, Post-Mortem, Tribal Knowledge |
| **Audit Orchestrator** | Monitors memory health | Gap Detector, Staleness, Single Point of Failure |

All requests flow through the **Master Orchestrator**, which classifies request type and delegates — mirroring how a real enterprise multi-agent system (like a company's internal AI assistant) routes work across specialised departments.

---

## 🤖 The Agents

### Retrieval Agents (answer questions)

| Agent | Purpose |
|---|---|
| **RouterAgent** | Classifies every incoming query into DECISION / PEOPLE / POLICY / PROJECT / UNKNOWN |
| **DecisionAgent** | Reconstructs why decisions were made, who made them, and the reasoning |
| **PeopleAgent** | Answers questions about individuals — roles, relationships, communication patterns |
| **PolicyAgent** | Explains company policies, procedures, and their rationale |
| **ProjectAgent** | Tracks project history, status, team, and outcomes |
| **CompetitiveAgent** | Combines internal memory with live web search for competitive intelligence |

### Capture Agents (ingest knowledge)

| Agent | Purpose |
|---|---|
| **MeetingAgent** | Extracts decisions, action items, owners, and deadlines from meeting transcripts |
| **PostMortemAgent** | Extracts what worked, what failed, root causes, and lessons learned from retrospectives |
| **TribalKnowledgeAgent** | Conducts structured interviews with experts to capture undocumented knowledge |

### Audit Agents (monitor health)

| Agent | Purpose |
|---|---|
| **GapDetectorAgent** | Finds departments, topics, and time periods with insufficient documentation |
| **StalenessAgent** | Detects outdated content, stale policies, and inactive contributors |
| **SinglePointOfFailureAgent** | Identifies individuals whose departure would cause irreplaceable knowledge loss |

**9 specialist agents. 3 sub-orchestrators. 1 master brain.**

---

## 🧬 Memory Architecture — Hybrid RAG

The system uses **two complementary memory stores**, not one:

```
┌─────────────────────┐         ┌─────────────────────┐
│      ChromaDB         │         │       Neo4j           │
│   (Vector Memory)      │         │   (Graph Memory)       │
├─────────────────────┤         ├─────────────────────┤
│ Semantic search over   │         │ Relationships:          │
│ email/document chunks  │         │  Person → MADE_DECISION │
│                        │         │  Person → INVOLVED_IN   │
│ "What was discussed     │         │  Person → COMMUNICATED  │
│  about the Austin trip?"│         │                        │
│                        │         │ "Who did X work with?"  │
│                        │         │ "What decisions has X   │
│                        │         │  been involved in?"     │
└─────────────────────┘         └─────────────────────┘
           │                               │
           └───────────────┬───────────────┘
                            ▼
                    MemoryManager
              (unified interface + cache)
```

- **ChromaDB** — semantic similarity search over document/email chunks (SentenceTransformers embeddings, cosine similarity)
- **Neo4j** — structured relationship graph connecting People → Decisions → Projects → Policies
- **In-memory LRU Cache** — avoids redundant embedding calls and LLM invocations for repeated queries

Every retrieval agent queries **both** stores and synthesises a single grounded answer.

---

## 🕸️ LangGraph State Machines

Instead of simple function chains, the system is built on **four compiled LangGraph state machines**:

| Graph | Flow |
|---|---|
| **Retrieval Graph** | `classify_query → route by category → retrieve → fallback (if low confidence) → respond` |
| **Capture Graph** | `validate input → route by content type → capture agent → respond` |
| **Audit Graph** | `initialise → gap scan → staleness scan → SPF scan → aggregate → respond` |
| **Master Graph** | `receive → route by request type → sub-orchestrator → respond` |

Each graph node returns a **partial state update**; conditional edges route execution dynamically based on classification confidence, content type, or audit scope — giving the system genuine branching, fallback, and self-correction behaviour rather than a fixed pipeline.

---

## 📂 Project Structure

```
institutional-memory-system/
│
├── .env                          # Environment secrets (not committed)
├── .env.example                  # Template for environment variables
├── .gitignore
├── requirements.txt
├── README.md
│
├── config/
│   ├── settings.py                # Pydantic-settings centralised configuration
│   └── logging_config.py          # Loguru structured logging setup
│
├── data/
│   ├── raw/                       # Raw Enron email CSV
│   ├── processed/                 # Cleaned email CSV output
│   └── synthetic/                 # Synthetic policy/decision data
│
├── schemas/
│   ├── email_schema.py            # RawEmail / CleanEmail / EmailChunk models
│   ├── agent_schema.py            # AgentInput / AgentOutput / QueryCategory
│   ├── query_schema.py            # API request/response models
│   └── memory_schema.py           # VectorSearchResult / graph node models
│
├── ingestion/
│   ├── email_parser.py            # Parses & cleans raw Enron emails
│   ├── chunker.py                 # Splits emails into overlapping chunks
│   ├── embedder.py                # Embeds chunks & stores in ChromaDB
│   ├── metadata_extractor.py      # Extracts intent, topics, action items
│   └── pipeline.py                # Orchestrates the full ingestion flow
│
├── memory/
│   ├── vector_store.py            # ChromaDB interface
│   ├── graph_store.py             # Neo4j interface
│   ├── cache.py                   # In-memory LRU query cache
│   └── memory_manager.py          # Unified memory interface (cache + vector + graph)
│
├── agents/
│   ├── base_agent.py              # Abstract base class all agents inherit
│   ├── retrieval/
│   │   ├── router_agent.py
│   │   ├── decision_agent.py
│   │   ├── people_agent.py
│   │   ├── policy_agent.py
│   │   ├── project_agent.py
│   │   └── competitive_agent.py
│   ├── capture/
│   │   ├── meeting_agent.py
│   │   ├── postmortem_agent.py
│   │   └── tribal_knowledge_agent.py
│   └── audit/
│       ├── gap_detector_agent.py
│       ├── staleness_agent.py
│       └── single_point_failure_agent.py
│
├── orchestrators/
│   ├── master_orchestrator.py     # Top-level request router
│   ├── retrieval_orchestrator.py  # Routes queries to specialist agents
│   ├── capture_orchestrator.py    # Routes content to capture agents
│   └── audit_orchestrator.py      # Coordinates all audit scans
│
├── graphs/
│   ├── states.py                  # LangGraph TypedDict state schemas
│   ├── nodes.py                   # All LangGraph node functions
│   ├── edges.py                   # All conditional edge/routing functions
│   └── memory_graph.py            # Compiles & wires all 4 LangGraph graphs
│
├── api/
│   ├── main.py                    # FastAPI app entry point
│   ├── middleware.py              # Request logging middleware
│   ├── dependencies.py            # Shared FastAPI dependency injection
│   └── routes/
│       ├── query.py                # POST /api/v1/query/
│       ├── ingest.py                # POST /api/v1/ingest/capture, /bulk
│       ├── audit.py                  # POST /api/v1/audit/, GET /quick
│       └── health.py                  # GET /api/v1/health/, /live
│
├── ui/
│   ├── app.py                     # Streamlit home page + global navy theme
│   ├── components/
│   │   ├── chat.py                 # Reusable chat interface component
│   │   └── source_viewer.py         # Source citation card component
│   └── pages/
│       ├── 01_query.py              # Ask questions interface
│       ├── 02_ingest.py              # Capture meetings/postmortems/tribal
│       ├── 03_audit.py                # Audit health dashboard
│       └── 04_graph_explorer.py        # Interactive Neo4j graph visualiser
│
└── tests/
    ├── unit/
    └── integration/
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Google Gemini 2.5 Flash (`gemini-2.5-flash`) |
| **Agent Framework** | LangChain 1.x + LangGraph 1.x |
| **Vector Database** | ChromaDB (persistent, cosine similarity) |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **Graph Database** | Neo4j |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | Streamlit (custom navy enterprise theme) |
| **Configuration** | Pydantic Settings |
| **Logging** | Loguru (structured JSON + console) |
| **Data Source** | Enron Email Dataset (Kaggle) |
| **Language** | Python 3.11+ |

---

## ⚙️ Setup Instructions

### 1. Clone & create virtual environment

```bash
python -m venv myvenv
myvenv\Scripts\activate        # Windows
source myvenv/bin/activate     # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=institutional_memory
EMBEDDING_MODEL=all-MiniLM-L6-v2

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

RAW_DATA_PATH=./data/raw/emails.csv
PROCESSED_DATA_PATH=./data/processed/emails_clean.csv
MAX_EMAILS=1000
```

### 4. Download the dataset

Download the [Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) from Kaggle and place `emails.csv` in `data/raw/`.

### 5. Run the ingestion pipeline

```bash
python -m ingestion.pipeline
```

This parses, cleans, chunks, embeds, and stores emails into ChromaDB.

### 6. (Optional) Start Neo4j

```bash
# Using Docker
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password neo4j:latest
```

> The system runs without Neo4j — graph features degrade gracefully.

### 7. Run the API

```bash
uvicorn api.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive Swagger documentation.

### 8. Run the UI

```bash
streamlit run ui/app.py
```

Visit `http://localhost:8501`.

---

## 🔌 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/query/` | Submit a natural language query |
| `POST` | `/api/v1/ingest/capture` | Capture a meeting, post-mortem, or tribal knowledge item |
| `POST` | `/api/v1/ingest/bulk` | Trigger bulk email ingestion (background task) |
| `POST` | `/api/v1/audit/` | Run a full or scoped audit scan |
| `GET` | `/api/v1/audit/quick` | Run a lightweight gap-only audit scan |
| `GET` | `/api/v1/health/` | Full subsystem health check |
| `GET` | `/api/v1/health/live` | Lightweight liveness probe |

### Example: Query request

```json
POST /api/v1/query/
{
  "query": "Why did we change our vendor selection process?",
  "top_k": 5
}
```

### Example: Capture request

```json
POST /api/v1/ingest/capture
{
  "content": "Q3 Budget Review — Decision: Cut marketing by 15%...",
  "content_type": "meeting_transcript"
}
```

---

## 🎨 UI Design System

Built with a custom navy enterprise theme inspired by Stripe, Linear, and Notion:

| Token | Value |
|---|---|
| Brand Navy | `#3A2F69` |
| Deep Background | `#120E28` |
| Surface | `#1C1740` / `#221C4D` |
| Accent | `#8B7FFF` |
| Success / Warning / Danger | `#34D399` / `#FBBF24` / `#F87171` |
| Typography | Inter (UI) + JetBrains Mono (data) |
| Radius | 10–14px |

Features: glassmorphism cards, soft shadows, hover micro-interactions, live status indicators, and an interactive `vis-network` graph visualiser.

---

## 🧪 Key Design Patterns Used

- **Abstract Base Agent** — all 12 agents inherit shared LLM invocation, retrieval, and output formatting from `BaseAgent`
- **Hybrid RAG** — every retrieval agent combines ChromaDB semantic search with Neo4j graph traversal
- **Fallback Routing** — low-confidence classifications automatically retry with a secondary agent
- **Graceful Degradation** — the system functions fully without Neo4j; graph features simply disable
- **LangGraph State Machines** — typed state, partial updates, conditional edges instead of linear chains
- **Cache-Aware Retrieval** — `MemoryManager` checks an LRU cache before hitting ChromaDB
- **Structured LLM Outputs** — all agents use strict prompt-enforced formats parsed via regex, not free-form text
- **Dependency Injection** — all agents/orchestrators accept an optional `MemoryManager` for clean testability

---

## 📊 Dataset

This project uses the **Enron Email Dataset** (~1000 emails subset) as realistic historical corporate communication data, simulating institutional memory for a real organisation — including real decisions, project discussions, and workplace dynamics.

---

## 🎯 What This Project Demonstrates

- Multi-agent orchestration with sub-orchestrators (not a single flat agent list)
- Production-grade LangGraph state machine design
- Hybrid vector + graph RAG architecture
- Enterprise-style FastAPI backend with middleware, DI, and structured logging
- Full-stack delivery: ingestion → agents → orchestration → API → custom UI
- Real-world problem framing: institutional knowledge loss in organisations

---

## 📄 License

Personal Project.
