# Lab 17 – Multi-Backend Memory Agent with LangGraph

This project implements a conversational agent with a **full 4-layer memory stack**, LangGraph state machine, context window management, and performance benchmarking.

## Architecture

```
User Input → [extract_memory] → [retrieve_memory] → [generate_response] → AI Output
                  ↓                     ↑
            Save facts/episodes    Query all 4 backends
            to Long-term/Episodic  (Short-term + Long-term + Episodic + Semantic)
```

## Memory Stack (4 backends)

| Memory Type | Backend | Purpose |
|-------------|---------|---------|
| **Short-term** | Sliding window buffer (`MemoryState.messages`) | Recent conversation context |
| **Long-term** | Redis (fallback: in-memory dict) | User profile (name, allergies, preferences) |
| **Episodic** | JSON file (`episodes.json`) | Significant events, lessons learned |
| **Semantic** | ChromaDB with all-MiniLM-L6-v2 embeddings | FAQ, knowledge base retrieval |

## Project Structure

- `agent/` — LangGraph workflow: `MemoryState`, `extract_memory`, `retrieve_memory`, `generate_response`
- `memory/` — Four memory backends + seed data for semantic memory
- `router/` — Intent classifier and memory routing logic
- `context_manager/` — Token-aware context window manager with priority-based auto-trimming
- `benchmark/` — 10 multi-turn test conversations + automated evaluation script

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your `GOOGLE_API_KEY`.
3. (Optional) Run a local Redis server for persistent long-term memory:
   ```bash
   docker run -p 6379:6379 -d redis
   ```

## Running the Agent

Interactive CLI mode:
```bash
python -m agent.agent
```

## Running the Benchmark

Compares stateless (no-memory) vs stateful (full-memory) agent on 10 multi-turn conversations:
```bash
python benchmark/run_benchmark.py
```
Generates `BENCHMARK.md` at project root.

## Key Features

- **LLM-based fact extraction** with conflict handling (overwrites old facts when user corrects)
- **Real vector embeddings** via ChromaDB + sentence-transformers
- **Token budget management** with `tiktoken` and 4-level priority auto-trimming
- **LangGraph state machine** with typed `MemoryState` and 3-node pipeline
