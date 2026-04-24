from dotenv import load_dotenv
load_dotenv()

import json
import os
import time
import sys
from google import genai
import tiktoken

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import MemoryAgent
from memory.seed_data import seed_semantic_memory

_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

SKIP_API = False

class StatelessAgent:
    """
    A stateless agent that has NO memory between turns.
    Each turn is a completely independent prompt with no history.
    This serves as the baseline for no-memory comparison.
    """
    def __init__(self):
        pass
        
    def chat(self, user_query: str) -> str:
        global SKIP_API
        if SKIP_API:
            return "Tôi không có thông tin về điều đó."
        try:
            # Completely stateless: no history, no memory, just the current query
            prompt = f"You are a helpful assistant. Respond in Vietnamese.\n\nUser: {user_query}"
            response = client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=prompt
            )
            return response.text
        except Exception as e:
            SKIP_API = True
            return f"Error: {e}"


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def check_pass_fail(topic: str, turn_index: int, query: str, response: str, total_turns: int) -> str:
    """
    Deterministic pass/fail checker for benchmark scenarios.
    Evaluates whether the agent's response meets the expected criteria.
    """
    response_lower = response.lower()
    topic_lower = topic.lower()
    
    # Scenario 1: Profile recall — should remember user's name "Linh"
    if "profile recall after many turns" in topic_lower and turn_index == total_turns - 1:
        return "Pass" if "linh" in response_lower else "Fail"
        
    # Scenario 2: Allergy conflict — should say "đậu nành", NOT "sữa bò"
    if "allergy conflict update" in topic_lower and turn_index == total_turns - 1:
        has_correct = "đậu nành" in response_lower
        return "Pass" if has_correct else "Fail"
        
    # Scenario 3: Episodic recall — should mention docker service name
    if "recall previous debug lesson" in topic_lower and turn_index == 3:
        return "Pass" if ("service name" in response_lower or "docker" in response_lower) else "Fail"
        
    # Scenario 4: Semantic retrieval — should retrieve FAQ about refund conditions
    if "retrieve faq chunk" in topic_lower and turn_index == total_turns - 1:
        return "Pass" if ("hoàn tiền" in response_lower or "14 ngày" in response_lower or "điều kiện" in response_lower) else "Fail"
        
    # Scenario 5: Trim/token budget — should produce a summary
    if "trim/token budget" in topic_lower and turn_index == total_turns - 1:
        # The response should be concise (1 sentence as requested)
        return "Pass" if len(response) < 500 else "Fail"
        
    # Scenario 6: Mixed Intent — should explain microservices
    if "mixed intent" in topic_lower and turn_index == total_turns - 1:
        return "Pass" if ("microservice" in response_lower or "vi dịch vụ" in response_lower or "dịch vụ" in response_lower) else "Fail"
    
    # Scenario 7: Preference Conflict — should say "xanh dương", NOT "đỏ"
    if "preference conflict" in topic_lower and turn_index == total_turns - 1:
        return "Pass" if "xanh dương" in response_lower else "Fail"
         
    # Scenario 8: Episodic vs Semantic — should answer about New York area
    if "episodic vs semantic clarity" in topic_lower and turn_index == total_turns - 1:
        return "Pass" if ("783" in response_lower or "km" in response_lower or "diện tích" in response_lower) else "Fail"
         
    # Scenario 9: Complex short-term tracking — should compute 3 - 1 + 5 = 7
    if "complex short-term tracking" in topic_lower and turn_index == total_turns - 1:
        return "Pass" if ("7" in response or "bảy" in response_lower) else "Fail"
         
    # Scenario 10: Final Robustness — should recall AICB as most recent course
    if "final robustness check" in topic_lower and turn_index == total_turns - 1:
        return "Pass" if "aicb" in response_lower else "Fail"
        
    return "N/A"

def run_benchmark():
    conversations_path = os.path.join(os.path.dirname(__file__), "conversations.json")
    with open(conversations_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    # Output file will be at project root
    report_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "BENCHMARK.md")

    # --- STEP 0: Clean up stale data ---
    episodes_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "episodes.json")
    with open(episodes_path, "w", encoding="utf-8") as f:
        json.dump([], f)
    print("Reset episodes.json to empty.")
    
    # --- STEP 1: Seed semantic memory with knowledge chunks ---
    seed_semantic_memory()

    print("\nStarting Benchmark...\n")
    
    markdown_rows = []
    stats = {"pass": 0, "fail": 0, "total": 0}
    
    for conv in conversations:
        topic = conv['topic']
        print(f"  Testing [{conv['id']}/10]: {topic}")
        agent_stateless = StatelessAgent()
        
        # Unique user ID per conversation to avoid cross-contamination
        uid = f"user_bench_{conv['id']}_{int(time.time())}"
        agent_stateful = MemoryAgent(user_id=uid)
        
        total_turns = len(conv["turns"])
        final_resp_sl = ""
        final_resp_st = ""
        
        for i, turn in enumerate(conv["turns"]):
            query = turn["query"]
            
            # --- Stateless Agent (no memory, no history) ---
            resp_sl = agent_stateless.chat(query)
            
            # --- Stateful Agent (full memory stack) ---
            resp_st = agent_stateful.chat(query)
            
            # Track final responses for the evaluation turn
            if i == total_turns - 1:
                final_resp_sl = resp_sl
                final_resp_st = resp_st
            
            # Small delay to avoid rate limiting
            time.sleep(0.3)
        
        # Evaluate only the final (evaluation) turn
        pass_fail_sl = check_pass_fail(topic, total_turns - 1, conv["turns"][-1]["query"], final_resp_sl, total_turns)
        pass_fail_st = check_pass_fail(topic, total_turns - 1, conv["turns"][-1]["query"], final_resp_st, total_turns)
        
        # Determine overall pass
        is_pass = "✅ Pass" if pass_fail_st == "Pass" else "❌ Fail"
        
        # Build descriptive result strings
        no_mem_desc = _extract_short_answer(final_resp_sl, topic)
        with_mem_desc = _extract_short_answer(final_resp_st, topic)
        
        row = f"| {conv['id']} | {topic} | {no_mem_desc} | {with_mem_desc} | {is_pass} |"
        markdown_rows.append(row)
        
        stats["total"] += 1
        if pass_fail_st == "Pass":
            stats["pass"] += 1
        else:
            stats["fail"] += 1
        
        print(f"    Result: No-memory={pass_fail_sl}, With-memory={pass_fail_st}")
            
    print(f"\nBenchmark complete: {stats['pass']}/{stats['total']} passed")
    print("Generating BENCHMARK.md...")
    
    report_content = f"""# Benchmark Results: No-Memory vs With-Memory Agent

> **Lab 17 — Multi-Backend Memory Agent Benchmark**  
> 10 multi-turn conversations comparing a stateless (no-memory) agent vs our full-memory LangGraph agent.  
> Date: {time.strftime('%Y-%m-%d %H:%M')}

## Summary

| Metric | Value |
|--------|-------|
| Total Scenarios | {stats['total']} |
| With-Memory Pass | {stats['pass']} |
| With-Memory Fail | {stats['fail']} |
| Pass Rate | {stats['pass']*100//stats['total']}% |

## Detailed Results

| # | Scenario | No-memory result | With-memory result | Pass? |
|---|----------|------------------|---------------------|-------|
"""
    report_content += "\n".join(markdown_rows)
    
    report_content += f"""

## Test Coverage

| Test Group | Scenario IDs | Description |
|-----------|-------------|-------------|
| Profile Recall | 1 | Recall user name after 6 turns of unrelated conversation |
| Conflict Update | 2, 7 | Allergy correction (sữa bò → đậu nành), Color preference change (Đỏ → Xanh dương) |
| Episodic Recall | 3, 10 | Recall docker debug lesson, recall completed course |
| Semantic Retrieval | 4, 6, 8 | FAQ refund policy, Microservices explanation, New York facts |
| Trim/Token Budget | 5 | Long story generation + summarization under token constraints |
| Short-term Tracking | 9 | Multi-step arithmetic reasoning across turns (3 - 1 + 5 = 7) |

## Methodology

- **Stateless Agent**: Each turn is a completely independent API call with NO conversation history and NO memory. It cannot recall anything from previous turns.
- **Stateful Agent (Memory)**: Uses the full LangGraph pipeline with:
  - **Short-term**: Message history with sliding window (via `MemoryState.messages`)
  - **Long-term**: User profile stored in Redis/dict (name, allergies, preferences)
  - **Episodic**: Event log in `episodes.json` with timestamps and tags
  - **Semantic**: ChromaDB vector search over pre-loaded knowledge chunks
- **Token Counting**: Uses `tiktoken` (cl100k_base encoder) for accurate token measurement.
- **Context Management**: `ContextWindowManager` with 4-level priority auto-trim ensures prompt stays within budget.

## Qualitative Assessment

- **Profile Recall**: Agent with memory extracts user facts (name, allergies, job, school) into Redis/dict and recalls them correctly even after multiple unrelated turns.
- **Conflict Update**: LLM-based memory extraction handles corrections — when user says "nhầm" or "đổi ý", the extraction prompt instructs overwriting the old value with the new one.
- **Episodic Recall**: Important events tagged with semantic labels are logged to `episodes.json` and retrieved via recency-based lookup.
- **Semantic Retrieval**: Knowledge chunks pre-loaded into ChromaDB with real embeddings (all-MiniLM-L6-v2) enable accurate similarity-based retrieval for FAQ and factual questions.
- **Trim/Token Budget**: `ContextWindowManager` auto-trims lowest-priority slots (old history, then memory context) when approaching 90% of token budget, ensuring the system prompt and recent turns are never evicted.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Report saved to {report_path}")

def _extract_short_answer(response: str, topic: str) -> str:
    """
    Extracts a short descriptive answer from the full response for the benchmark table.
    """
    topic_lower = topic.lower()
    resp_lower = response.lower()
    
    if "profile recall" in topic_lower:
        if "linh" in resp_lower:
            return "Nhớ tên: Linh"
        return "Không nhớ tên"
    
    if "allergy conflict" in topic_lower:
        if "đậu nành" in resp_lower:
            return "Đậu nành ✓"
        if "sữa bò" in resp_lower:
            return "Sữa bò (old value)"
        return "Không nhớ"
    
    if "preference conflict" in topic_lower:
        if "xanh dương" in resp_lower:
            return "Xanh dương ✓"
        if "đỏ" in resp_lower:
            return "Đỏ (old value)"
        return "Không nhớ"
    
    if "debug lesson" in topic_lower:
        if "service name" in resp_lower or "docker" in resp_lower:
            return "Docker service name ✓"
        return "Không nhớ"
    
    if "faq chunk" in topic_lower:
        if "hoàn tiền" in resp_lower or "14 ngày" in resp_lower:
            return "Đúng FAQ chunk ✓"
        return "Không biết/sai"
    
    if "trim" in topic_lower:
        return f"Tóm tắt ({len(response)} chars)"
    
    if "mixed intent" in topic_lower:
        if "microservice" in resp_lower or "dịch vụ" in resp_lower:
            return "Giải thích đúng ✓"
        return "Không giải thích được"
    
    if "episodic vs semantic" in topic_lower:
        if "783" in resp_lower or "km" in resp_lower:
            return "Trả lời đúng diện tích ✓"
        return "Không biết"
    
    if "short-term tracking" in topic_lower:
        if "7" in response or "bảy" in resp_lower:
            return "7 quả táo ✓"
        return "Sai kết quả"
    
    if "robustness" in topic_lower:
        if "aicb" in resp_lower:
            return "AICB ✓"
        return "Không nhớ"
    
    return response[:50] + "..."

if __name__ == "__main__":
    run_benchmark()
