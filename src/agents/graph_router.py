from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from src.retrieval.hybrid_search import HybridRetriever


class GraphState(TypedDict):
    query: str
    chat_history: List[Dict[str, str]]  # Format: [{"role": "user"|"assistant", "content": "..."}]
    route: str
    documents: List[Dict[str, Any]]
    answer: str


llm = Ollama(model="llama3.2")
retriever = HybridRetriever()


def format_history(history: List[Dict[str, str]]) -> str:
    """Helper to convert structured message list into a plain text block."""
    if not history:
        return "No prior context."
    formatted = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)


def route_query_node(state: GraphState) -> GraphState:
    query = state["query"]
    history_str = format_history(state.get("chat_history", []))

    prompt = (
        f"Conversation History:\n{history_str}\n\n"
        f"Latest User Query: {query}\n\n"
        f"Analyze the latest query in context of the conversation history. "
        f"Decide if answering requires external document retrieval or direct generation.\n"
        f"Respond with EXACTLY one word: 'retrieval' or 'direct'.\n"
        f"Decision:"
    )
    response = llm.invoke(prompt).strip().lower()
    route = "retrieval" if "retrieval" in response else "direct"
    return {**state, "route": route}


def retrieve_context_node(state: GraphState) -> GraphState:
    query = state["query"]
    docs = retriever.search(query=query, top_k=3)
    return {**state, "documents": docs}


def generate_rag_node(state: GraphState) -> GraphState:
    query = state["query"]
    history_str = format_history(state.get("chat_history", []))
    docs = state.get("documents", [])

    context_str = (
        "\n\n".join([f"Document {i+1}:\n{doc['text']}" for i, doc in enumerate(docs)])
        if docs
        else "No relevant documents found."
    )

    prompt = (
        f"You are a helpful assistant. Use the conversation history and context below to answer.\n\n"
        f"Conversation History:\n{history_str}\n\n"
        f"Context:\n{context_str}\n\n"
        f"Latest User Query: {query}\n"
        f"Answer:"
    )
    answer = llm.invoke(prompt)
    return {**state, "answer": answer}


def generate_direct_node(state: GraphState) -> GraphState:
    query = state["query"]
    history_str = format_history(state.get("chat_history", []))

    prompt = (
        f"You are a helpful assistant. Answer the user prompt considering the conversation history.\n\n"
        f"Conversation History:\n{history_str}\n\n"
        f"Latest User Query: {query}\n"
        f"Answer:"
    )
    answer = llm.invoke(prompt)
    return {**state, "answer": answer}


def decide_next_node(state: GraphState) -> str:
    return state["route"]


# Build Graph
workflow = StateGraph(GraphState)

workflow.add_node("router", route_query_node)
workflow.add_node("retrieve", retrieve_context_node)
workflow.add_node("generate_rag", generate_rag_node)
workflow.add_node("generate_direct", generate_direct_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    decide_next_node,
    {
        "retrieval": "retrieve",
        "direct": "generate_direct",
    },
)

workflow.add_edge("retrieve", "generate_rag")
workflow.add_edge("generate_rag", END)
workflow.add_edge("generate_direct", END)

app = workflow.compile()