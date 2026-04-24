import json
import os
from google import genai
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

from agent.state import MemoryState
from memory.long_term import save_long_term, load_long_term
from memory.episodic import log_episode, get_recent_episodes
from memory.semantic import query_semantic
from context_manager.window_manager import ContextWindowManager

# Initialize Google AI Client
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def call_gemini(prompt: str, json_mode: bool = False) -> str:
    """Helper to call Gemini model and return text response."""
    from google.genai import types
    config = types.GenerateContentConfig(
        response_mime_type="application/json" if json_mode else "text/plain",
        temperature=0 if json_mode else 0.7
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=config
    )
    return response.text

def _parse_json_safe(raw: str) -> dict:
    """
    Safely parse JSON from LLM output, handling common formatting issues.
    Strips markdown code fences and retries on failure.
    """
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # Remove first ```json line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object from the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {}

def extract_memory_node(state: MemoryState) -> dict:
    """
    LLM-based extraction node: analyzes the latest user message to extract
    facts (for long-term profile) and episodic events (for episodic memory).
    Implements conflict handling by instructing LLM to overwrite old facts.
    """
    latest_msg = state["messages"][-1].content
    user_id = state["user_id"]
    
    current_profile = load_long_term(user_id)
    
    prompt = f"""You are a memory extraction assistant. Analyze the user message and extract structured information.

USER MESSAGE: "{latest_msg}"

CURRENT USER PROFILE: {json.dumps(current_profile, ensure_ascii=False)}

INSTRUCTIONS:
1. **FACT EXTRACTION (Long-term Profile)**:
   - Extract personal facts: name, job, school, preferences, allergies, favorite things, etc.
   - CRITICAL CONFLICT HANDLING: If the user CORRECTS or CHANGES a previous fact (e.g., "nhầm", "đổi ý", "không phải ... mà là ..."), you MUST:
     a) Set the NEW value for that key
     b) Add a "_deleted" entry to explicitly mark the old value as removed
   - Example: If profile has {{"allergy": "sữa bò"}} and user says "nhầm, tôi dị ứng đậu nành chứ không phải sữa bò",
     return {{"facts": {{"allergy": "đậu nành"}}}}
   - Example: If profile has {{"favorite_color": "Đỏ"}} and user says "đổi ý rồi, giờ thích Xanh dương",
     return {{"facts": {{"favorite_color": "Xanh dương"}}}}

2. **EPISODIC LOGGING**:
   - Extract significant events, experiences, lessons learned, or explicit instructions to remember something.
   - Include specific details, not vague summaries.
   - If user says "nhớ nhé" or "lưu lại", that's an explicit instruction to log an episode.

OUTPUT FORMAT (valid JSON only):
{{
    "facts": {{"key": "value"}},
    "episode": {{"event": "description of event", "tags": ["tag1", "tag2"]}}
}}

If nothing to extract for a section, use empty object/null:
{{
    "facts": {{}},
    "episode": null
}}"""
    
    try:
        raw = call_gemini(prompt, json_mode=True)
        extraction = _parse_json_safe(raw)
        
        facts = extraction.get("facts", {})
        if facts:
            for k, v in facts.items():
                if not k.startswith("_"):  # Skip meta keys
                    save_long_term(user_id, k, v)
                
        episode = extraction.get("episode")
        if episode and isinstance(episode, dict) and episode.get("event"):
            log_episode(episode["event"], episode.get("tags", []), filepath="episodes.json")
                
    except Exception as e:
        print(f"Error extracting memory: {e}")
        
    return {}

def retrieve_memory_node(state: MemoryState) -> dict:
    """
    Retrieves memories from all backends to populate the state for prompt injection.
    This is the memory router — it gathers context from all 4 memory types.
    """
    user_id = state["user_id"]
    latest_msg = state["messages"][-1].content
    
    # 1. Profile (Long-term / Redis)
    profile = load_long_term(user_id)
    
    # 2. Episodic (JSON file)
    episodes = get_recent_episodes(filepath="episodes.json", limit=5)
    
    # 3. Semantic (ChromaDB vector search)
    semantic_res = query_semantic(latest_msg, n_results=3)
    semantic_hits = semantic_res.get("documents", [[]])[0] if semantic_res else []
    
    return {
        "user_profile": profile,
        "episodes": episodes,
        "semantic_hits": semantic_hits
    }

def generate_response_node(state: MemoryState) -> dict:
    """
    Generates the AI response by injecting all retrieved memories into the prompt.
    Uses ContextWindowManager for token budget management and auto-trimming.
    """
    # Get the latest user message for response generation
    latest_msg = state["messages"][-1].content
    
    window_manager = ContextWindowManager(max_tokens=state.get("memory_budget", 4000))
    
    # P1: System prompt (never trimmed)
    window_manager.slots["system"]["content"] = (
        "You are a helpful, intelligent assistant with a robust multi-backend memory system. "
        "You have access to the user's profile, past episodes, and a knowledge base. "
        "Use these memories to provide personalized, contextually-aware responses. "
        "When the user asks about something stored in their profile or episodes, reference it directly. "
        "Always respond in the same language the user is using."
    )
    
    # P3: Memory context injection — the core of prompt injection
    memory_parts = []
    if state.get("user_profile"):
        profile_str = json.dumps(state['user_profile'], ensure_ascii=False)
        memory_parts.append(f"[User Profile]: {profile_str}")
    if state.get("episodes"):
        episodes_str = json.dumps(state['episodes'], ensure_ascii=False, indent=1)
        memory_parts.append(f"[Recent Episodes]: {episodes_str}")
    if state.get("semantic_hits"):
        memory_parts.append(f"[Semantic Knowledge]: " + " | ".join(state['semantic_hits']))
        
    window_manager.slots["memory_context"]["content"] = "\n".join(memory_parts)
    
    # P2: Recent conversation history (short-term context)
    recent_msgs = state["messages"][:-1]
    history_str = "\n".join([f"{m.type}: {m.content}" for m in recent_msgs[-6:]])
    window_manager.slots["recent_turns"]["content"] = f"[Recent Conversation History]:\n{history_str}"
    
    # Build context with auto-trimming
    full_system_context = window_manager.build_context()
    
    # Build single prompt with system context + user message
    full_prompt = f"{full_system_context}\n\nUser: {latest_msg}"
    
    try:
        ai_response = call_gemini(full_prompt, json_mode=False)
    except Exception as e:
        ai_response = f"Error generating response: {e}"
        
    return {"messages": [("ai", ai_response)]}

# Build the LangGraph workflow
def build_graph():
    """
    Constructs the LangGraph state machine with 3 nodes:
    1. extract_memory — LLM-based fact/episode extraction from user input
    2. retrieve_memory — Gathers context from all 4 memory backends
    3. generate_response — Builds prompt with injected memory and generates reply
    """
    workflow = StateGraph(MemoryState)
    
    # Add nodes
    workflow.add_node("extract_memory", extract_memory_node)
    workflow.add_node("retrieve_memory", retrieve_memory_node)
    workflow.add_node("generate_response", generate_response_node)
    
    # Define edges: extract -> retrieve -> generate -> END
    workflow.set_entry_point("extract_memory")
    workflow.add_edge("extract_memory", "retrieve_memory")
    workflow.add_edge("retrieve_memory", "generate_response")
    workflow.add_edge("generate_response", END)
    
    return workflow.compile()
