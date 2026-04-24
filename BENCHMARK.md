# Benchmark Results: No-Memory vs With-Memory Agent

> **Lab 17 — Multi-Backend Memory Agent Benchmark**  
> 10 multi-turn conversations comparing a stateless (no-memory) agent vs our full-memory LangGraph agent.  
> Date: 2026-04-24 15:24

## Summary

| Metric | Value |
|--------|-------|
| Total Scenarios | 10 |
| With-Memory Pass | 10 |
| With-Memory Fail | 0 |
| Pass Rate | 100% |

## Detailed Results

| # | Scenario | No-memory result | With-memory result | Pass? |
|---|----------|------------------|---------------------|-------|
| 1 | Profile recall after many turns | Không nhớ tên | Nhớ tên: Linh | ✅ Pass |
| 2 | Allergy conflict update | Không nhớ | Đậu nành ✓ | ✅ Pass |
| 3 | Recall previous debug lesson (Episodic) | Docker service name ✓ | Docker service name ✓ | ✅ Pass |
| 4 | Retrieve FAQ chunk (Semantic) | Đúng FAQ chunk ✓ | Đúng FAQ chunk ✓ | ✅ Pass |
| 5 | Trim/token budget test | Tóm tắt (131 chars) | Tóm tắt (222 chars) | ✅ Pass |
| 6 | Mixed Intent & Identity | Giải thích đúng ✓ | Giải thích đúng ✓ | ✅ Pass |
| 7 | Preference Conflict | Xanh dương ✓ | Xanh dương ✓ | ✅ Pass |
| 8 | Episodic vs Semantic clarity | Trả lời đúng diện tích ✓ | Trả lời đúng diện tích ✓ | ✅ Pass |
| 9 | Complex Short-term tracking | Sai kết quả | 7 quả táo ✓ | ✅ Pass |
| 10 | Final Robustness Check | Không nhớ | AICB ✓ | ✅ Pass |

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
