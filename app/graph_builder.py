from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from src.config import settings
from src.retrieval.hybrid_search import HybridRetriever

class GraphState(TypedDict):
    query: str
    chat_history: List[Dict[str, str]]
    route: str
    documents: List[Dict[str, Any]]
    answer: str

# Initialize shared components using the settings instance
llm = OllamaLLM(model=settings.LLM_MODEL)
retriever = HybridRetriever()

def format_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return "No prior context."
    return "\n".join([f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" for msg in history])

def route_query_node(state: GraphState) -> GraphState:
    query = state["query"]
    history_str = format_history(state.get("chat_history", []))
    prompt = (
        f"Conversation History:\n{history_str}\n\n"
        f"Latest Query: {query}\n\n"
        f"Decide if answering this query requires document retrieval or direct generation.\n"
        f"Respond with EXACTLY one word: 'retrieval' or 'direct'.\n"
        f"Decision:"
    )
    try:
        response = llm.invoke(prompt).strip().lower()
        route = "retrieval" if "retrieval" in response else "direct"
    except Exception:
        route = "retrieval"
        
    return {**state, "route": route}

def retrieve_context_node(state: GraphState) -> GraphState:
    query = state["query"]
    docs = retriever.search(query=query, top_k=3)
    return {**state, "documents": docs}

def generate_rag_node(state: GraphState) -> GraphState:
    query = state["query"]
    history_str = format_history(state.get("chat_history", []))
    docs = state.get("documents", [])
    
    context_str = "\n\n".join([f"Document {i+1}:\n{doc['text']}" for i, doc in enumerate(docs)]) if docs else "No specific context found."
        
    prompt = (
        f"Conversation History:\n{history_str}\n\n"
        f"Context:\n{context_str}\n\n"
        f"Query: {query}\n\n"
        f"Provide a clear, accurate answer based on the context provided:\nAnswer:"
    )
    try:
        answer = llm.invoke(prompt)
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"
        
    return {**state, "answer": answer}

def generate_direct_node(state: GraphState) -> GraphState:
    query = state["query"]
    history_str = format_history(state.get("chat_history", []))
    prompt = f"Conversation History:\n{history_str}\n\nQuery: {query}\n\nAnswer:"
    
    try:
        answer = llm.invoke(prompt)
    except Exception as e:
        answer = f"Error generating response: {str(e)}"
        
    return {**state, "answer": answer}

def create_workflow() -> StateGraph:
    workflow = StateGraph(GraphState)
    
    workflow.add_node("router", route_query_node)
    workflow.add_node("retrieve", retrieve_context_node)
    workflow.add_node("generate_rag", generate_rag_node)
    workflow.add_node("generate_direct", generate_direct_node)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "retrieval": "retrieve",
            "direct": "generate_direct"
        }
    )
    workflow.add_edge("retrieve", "generate_rag")
    workflow.add_edge("generate_rag", END)
    workflow.add_edge("generate_direct", END)
    
    return workflow
def generate_answer_node(state: dict) -> dict:
    query = state.get("query", "")
    documents = state.get("documents", [])
    
    # Format retrieved context
    context = "\n\n".join([doc.get("text", "") for doc in documents])
    
    prompt = f"""You are an enterprise AI assistant. Answer the user question based on the provided context.

Context:
{context}

Question:
{query}

Answer:"""

    try:
        response = llm.invoke(prompt)
        state["answer"] = response
    except Exception as e:
        state["answer"] = f"Error generating answer: {str(e)}"
        
    return state

def generate_node(state: dict) -> dict:
    query = state.get("query", "")
    docs = state.get("documents", [])
    
    context = "\n\n".join([d.get("text", "") for d in docs])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    try:
        response = llm.invoke(prompt)
        state["answer"] = response
    except Exception as e:
        state["answer"] = f"Error generating answer: {str(e)}"
        
    return state