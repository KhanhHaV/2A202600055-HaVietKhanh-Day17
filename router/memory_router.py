import json
from enum import Enum
from memory.long_term import load_long_term
from memory.episodic import get_recent_episodes
from memory.semantic import query_semantic

class MemoryType(Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"

def classify_intent(query: str) -> MemoryType:
    """
    Classifies the intent of a query to determine the appropriate memory type.
    Uses keyword matching as a baseline. For a more robust solution, an LLM could be used here.
    """
    preference_keywords = ["thích", "yêu thích", "prefer", "favorite", "sở thích", "tôi là", "tên tôi"]
    experience_keywords = ["hôm qua", "lần trước", "trước đây", "đã nói", "last time", "hôm kia", "gần đây"]
    
    query_lower = query.lower()
    
    if any(kw in query_lower for kw in preference_keywords):
        return MemoryType.LONG_TERM
    elif any(kw in query_lower for kw in experience_keywords):
        return MemoryType.EPISODIC
    else:
        # For general factual questions, fallback to semantic search
        return MemoryType.SEMANTIC

def route_memory(query: str, user_id: str, short_term_memory=None) -> str:
    """
    Routes the query to the appropriate memory backend and retrieves context.
    """
    intent = classify_intent(query)
    
    if intent == MemoryType.LONG_TERM:
        data = load_long_term(user_id)
        if data:
            return f"User preferences: {json.dumps(data, ensure_ascii=False)}"
        return "No user preferences found."
        
    elif intent == MemoryType.EPISODIC:
        recent = get_recent_episodes(limit=5)
        if recent:
            return f"Recent episodes:\n{json.dumps(recent, ensure_ascii=False, indent=2)}"
        return "No recent episodes found."
        
    elif intent == MemoryType.SEMANTIC:
        results = query_semantic(query, n_results=3)
        docs = results.get("documents", [[]])[0]
        if docs:
            return "Semantic facts:\n" + "\n".join(docs)
        return "No semantic facts found."
        
    else:  # SHORT_TERM (Fallback if explicitly called or default)
        if short_term_memory:
            return short_term_memory.load_memory_variables({})["chat_history"]
        return "No short term context."
