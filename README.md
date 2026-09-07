# Enterprise Hybrid RAG & Multi-Agent Orchestration System

An async Python microservice featuring deterministic multi-agent state routing, hybrid document retrieval (BM25 + Qdrant Dense Vector Search), and automated evaluation pipelines for enterprise knowledge systems.

---

## Technical Overview

* **Deterministic Multi-Agent State Machine**: Orchestrates complex user requests across specialized Retrieval and Action agents using **LangGraph**.
* **Hybrid Search Engine**: Fuses sparse BM25 keyword rankings with dense Qdrant vector embeddings, rescored via Cross-Encoder re-ranking.
* **Async Microservice Architecture**: Native **FastAPI** REST endpoints with strict Pydantic input/output contracts.
* **Automated Evaluation Benchmarking**: Integrated **Ragas** testing suite measuring faithfulness, answer relevancy, and context recall.

---

## Technology Stack

| Layer | Component | Function |
| --- | --- | --- |
| **API / Service** | FastAPI + Uvicorn | Async REST endpoints & OpenAPI documentation |
| **Orchestration** | LangGraph + LangChain | Stateful multi-agent routing & tool execution |
| **Vector Indexing** | Qdrant | High-performance dense vector storage & payload filtering |
| **Keyword Search** | Rank-BM25 | Lexical text search for domain-specific terminology |
| **Embedding Engine** | `BAAI/bge-small-en-v1.5` | High-accuracy lightweight dense representations |
| **Evaluation** | Ragas | Continuous quantitative evaluation of LLM generation |

---

## Directory Architecture

```text
enterprise-rag-agent-system/
├── docs/                 # System architecture diagrams and API specs
├── evaluation/           # Ragas evaluation scripts & ground truth datasets
├── src/                  # Core Python source files
│   ├── agents/           # LangGraph state nodes & multi-agent routers
│   ├── api/              # FastAPI endpoints & Pydantic request models
│   ├── retrieval/        # Hybrid search, chunking, & Qdrant vector store drivers
│   └── tools/            # Agent tools (API connectors, document parsers)
├── tests/                # Unit & integration test suites
├── .gitignore
├── README.md             # Project documentation
└── requirements.txt      # Dependency manifest
Getting Started
Prerequisites
Python 3.10 or higher

Docker (Optional for local Qdrant instance)
Installation & Setup
Clone the repository:
git clone [https://github.com/sach7742/enterprise-rag-agent-system.git](https://github.com/sach7742/enterprise-rag-agent-system.git)
cd enterprise-rag-agent-system
Set up virtual environment:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Launch Qdrant Vector Database (Docker):
docker run -p 6333:6333 qdrant/qdrant
Run Application API:
uvicorn src.api.main:app --reload
