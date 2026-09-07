from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.graph_builder import create_workflow

app = FastAPI(title="Enterprise RAG Agent System")

# Compile LangGraph graph
workflow = create_workflow()
graph = workflow.compile()

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str, str]]] = []

class QueryResponse(BaseModel):
    query: str
    route: Optional[str] = "direct"
    answer: str
    documents: Optional[List[Dict[str, Any]]] = []

@app.get("/")
def read_root():
    return {"status": "online", "system": "Enterprise RAG Agent"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        initial_state = {
            "query": request.query,
            "chat_history": request.chat_history or [],
            "route": "",
            "documents": [],
            "answer": ""
        }
        # Invoke compiled workflow synchronously/asynchronously
        result = await graph.ainvoke(initial_state)
        return QueryResponse(
            query=result["query"],
            route=result.get("route", "direct"),
            answer=result.get("answer", "No response generated."),
            documents=result.get("documents", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))