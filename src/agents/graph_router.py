import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
# Replace DuckDuckGoSearchRun with DuckDuckGoSearchResults
from langchain_community.tools import DuckDuckGoSearchResults
from src.retrieval.hybrid_search import HybridRetriever


class GraphState(TypedDict):
    query: str
    chat_history: List[Dict[str, str]]  # Format: [{"role": "user"|"assistant", "content": "..."}]
    route: str
    documents: List[Dict[str, Any]]
    web_search_needed: bool
    answer: str
    sources: List[Dict[str, str]]  # <--- Added to hold metadata: [{"title": ..., "url": ...}]


llm = Ollama(model="llama3.2")
retriever = HybridRetriever()

# Initialize search tool configured to return JSON with titles & links
search_tool = DuckDuckGoSearchResults(output_format="json")


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
    return {**state, "route": route, "sources": []}


def retrieve_context_node(state: GraphState) -> GraphState:
    query = state["query"]
    docs = retriever.search(query=query, top_k=3)
    
    # Extract local document source links if present in metadata
    local_sources = []
    for d in docs:
        if isinstance(d, dict) and "metadata" in d and "source" in d["metadata"]:
            local_sources.append({
                "title": d["metadata"].get("title", "Local Document"),
                "url": d["metadata"]["source"]
            })

    return {**state, "documents": docs, "sources": local_sources}


def grade_documents_node(state: GraphState) -> GraphState:
    """Grades retrieved context using Llama 3.2 to verify relevance."""
    query = state["query"]
    docs = state.get("documents", [])

    if not docs:
        print("---RETRIEVAL EMPTY: TRIGGERING WEB SEARCH FALLBACK---")
        return {**state, "web_search_needed": True}

    context_str = "\n\n".join([f"Document {i+1}:\n{doc.get('text', '')}" for i, doc in enumerate(docs)])

    prompt = (
        f"User Query: {query}\n\n"
        f"Retrieved Documents:\n{context_str}\n\n"
        f"Are these documents relevant and sufficient to accurately answer the user's query?\n"
        f"Respond with EXACTLY one word: 'yes' or 'no'.\n"
        f"Assessment:"
    )
    response = llm.invoke(prompt).strip().lower()

    if "yes" in response:
        print("---RETRIEVAL RELEVANT: PROCEEDING TO GENERATE RAG---")
        return {**state, "web_search_needed": False}
    else:
        print("---RETRIEVAL IRRELEVANT: TRIGGERING WEB SEARCH FALLBACK---")
        return {**state, "web_search_needed": True}


def web_search_node(state: GraphState) -> GraphState:
    """Executes external web search fallback when local context grading fails."""
    query = state["query"]
    print(f"---EXECUTING WEB SEARCH FOR: '{query}'---")
    
    web_results_text = ""
    sources = []

    try:
        raw_results = search_tool.invoke(query)
        results = json.loads(raw_results) if isinstance(raw_results, str) else raw_results

        for item in results:
            title = item.get("title", "Web Source")
            url = item.get("link", item.get("href", ""))
            snippet = item.get("snippet", item.get("body", ""))

            web_results_text += f"Source Title: {title}\nURL: {url}\nSnippet: {snippet}\n\n"

            if url:
                sources.append({"title": title, "url": url})

        web_docs = [{"text": web_results_text, "score": 1.0, "metadata": {"source": "web"}}]

    except Exception as e:
        print(f"Web search error: {e}")
        web_docs = [{"text": "Web search fallback failed to retrieve online context.", "score": 0.0, "metadata": {}}]

    return {**state, "documents": web_docs, "sources": sources}


def generate_rag_node(state: GraphState) -> GraphState:
    query = state["query"]
    history_str = format_history(state.get("chat_history", []))
    docs = state.get("documents", [])

    context_str = (
        "\n\n".join([f"Document {i+1}:\n{doc.get('text', '')}" for i, doc in enumerate(docs)])
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
    return {**state, "answer": answer, "sources": []}


def decide_route(state: GraphState) -> str:
    return state["route"]


def decide_after_grade(state: GraphState) -> str:
    if state.get("web_search_needed"):
        return "web_search"
    return "generate_rag"


# Build Graph
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("router", route_query_node)
workflow.add_node("retrieve", retrieve_context_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate_rag", generate_rag_node)
workflow.add_node("generate_direct", generate_direct_node)

workflow.set_entry_point("router")

# Router Edge
workflow.add_conditional_edges(
    "router",
    decide_route,
    {
        "retrieval": "retrieve",
        "direct": "generate_direct",
    },
)

# Retrieval -> Grade -> Branch Sequence
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_after_grade,
    {
        "web_search": "web_search",
        "generate_rag": "generate_rag",
    },
)

# Connect Fallback & Generation to End
workflow.add_edge("web_search", "generate_rag")
workflow.add_edge("generate_rag", END)
workflow.add_edge("generate_direct", END)

app = workflow.compile()