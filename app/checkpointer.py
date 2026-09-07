from langgraph.checkpoint.memory import MemorySaver

def get_checkpointer() -> MemorySaver:
    """
    Returns an in-memory checkpointer for LangGraph state management.
    Prevents asyncio event loop crashes during imports on Python 3.14.
    """
    return MemorySaver()