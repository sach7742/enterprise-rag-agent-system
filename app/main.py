from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel

# Import the compiled graph from graph_router.py
# (Adjust the module path if graph_router.py is located elsewhere, e.g., src.graph_router)
from src.agents.graph_router import app as app_graph

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    title: str
    url: str


class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[SourceItem]] = []


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    # Align dictionary key with GraphState field ('query')
    initial_state = {"query": request.question}

    # Execute graph execution
    final_state = await app_graph.ainvoke(initial_state)

    return QueryResponse(
        answer=final_state.get("answer", "No answer generated."),
        sources=final_state.get("sources", []),
    )