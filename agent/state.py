from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph.message import add_messages

class MemoryState(TypedDict):
    """
    Represents the state of the Memory Agent across the LangGraph nodes.
    """
    # Chat history, uses add_messages reducer to append new messages
    messages: Annotated[list, add_messages]
    
    # Current user ID
    user_id: str
    
    # Retrieved memories to be injected into the prompt
    user_profile: dict
    episodes: List[dict]
    semantic_hits: List[str]
    
    # Token management
    memory_budget: int
