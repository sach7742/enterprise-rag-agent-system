# Enterprise RAG Agent System 🚀

An end-to-end, enterprise-grade Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **Streamlit**, **LangGraph**, and **Qdrant**. The system uses adaptive agentic routing to seamlessly process enterprise documents, generate hybrid search indexes, and answer queries with accurate source attribution.

---

## 🌟 Key Features

* **Agentic Graph Routing**: Built on **LangGraph** to dynamically route queries between direct LLM reasoning, document retrieval, and specialized tools.
* **Hybrid Search Retrieval**: Combines **Qdrant** vector database embeddings with **BM25** sparse retrieval for optimal document recall.
* **FastAPI Backend**: Async, production-ready REST API serving high-throughput inference endpoints.
* **Streamlit Interactive UI**: Simple, intuitive chat and administration dashboard for user interaction and document uploads.
* **Automated Data Ingestion**: Built-in pipeline for processing, chunking, and indexing unstructured documents.

---

## 📁 Repository Structure

```text
enterprise-rag-agent-system/
├── app/
│   └── main.py              # FastAPI application entrypoint
├── src/
│   ├── agent/               # LangGraph agent definitions & graph router
│   ├── agents/              # Core routing logic & state management
│   └── retrieval/           # Document ingestion, vector DB, & BM25 indexing
├── app_ui.py                # Streamlit frontend dashboard entrypoint
├── run.sh                   # Unified startup script (FastAPI + Streamlit)
├── requirements.txt         # Dependencies
└── README.md                # Project documentation

```

---

## ⚡ Quickstart

### Prerequisites

* **Python**: `3.10` or higher (Python `3.11` / `3.12` recommended)
* **Git**

### Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/sach7742/enterprise-rag-agent-system.git](https://github.com/sach7742/enterprise-rag-agent-system.git)
cd enterprise-rag-agent-system

```


2. **Create and activate a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Set up Environment Variables:**
Create a `.env` file in the root directory and add your required API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333

```



---

## 🚀 Running the Application

### Option 1: One-Command Start (Recommended)

Run both the FastAPI backend and Streamlit frontend together using `run.sh`:

```bash
chmod +x run.sh
./run.sh

```

* **FastAPI Backend**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **API Documentation (Swagger UI)**: [http://127.0.0.1:8000/docs](https://www.google.com/search?q=http://127.0.0.1:8000/docs)
* **Streamlit Web UI**: [http://127.0.0.1:8501](http://127.0.0.1:8501)

---

### Option 2: Running Services Separately

If you prefer to run services in separate terminal windows:

**1. Start the FastAPI Backend:**

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

```

**2. Start the Streamlit Frontend:**

```bash
streamlit run app_ui.py --server.port 8501

```

---

## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **Backend**: FastAPI, Uvicorn
* **Orchestration / Agents**: LangGraph, LangChain
* **Vector Database**: Qdrant
* **Keyword Search**: BM25 (Rank-BM25)



