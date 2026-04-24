# Lab 17 – Multi-Backend Memory Agent
> **Mục tiêu:** Xây dựng một conversational agent có hệ thống memory đa tầng, router thông minh, quản lý context window, và benchmark hiệu năng.

---

## 📋 Tổng quan bài tập

Bài lab yêu cầu hoàn thành **4 phần chính** và nộp kết quả dưới dạng **GitHub repo + benchmark report**. Điểm tối đa đạt được khi đáp ứng đầy đủ tất cả các tiêu chí dưới đây.

---

## Phần 1 – Implement 4 Memory Backends

Triển khai bốn loại memory riêng biệt, mỗi loại phục vụ một mục đích khác nhau.

### 1.1 ConversationBufferMemory (Short-term)

- **Mục đích:** Lưu các lượt hội thoại gần nhất trong phiên hiện tại.
- **Yêu cầu:**
  - Sử dụng `ConversationBufferMemory` từ LangChain (hoặc tự implement tương đương).
  - Lưu toàn bộ lịch sử hội thoại dưới dạng list các cặp `(human, ai)`.
  - Hỗ trợ reset sau mỗi phiên.

```python
from langchain.memory import ConversationBufferMemory

short_term = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

### 1.2 Redis (Long-term)

- **Mục đích:** Lưu trữ thông tin người dùng bền vững qua nhiều phiên.
- **Yêu cầu:**
  - Kết nối Redis (local hoặc Redis Cloud).
  - Lưu các fact về user theo `user_id` (tên, sở thích, lịch sử quan trọng).
  - Có TTL (time-to-live) hợp lý hoặc persistent.

```python
import redis
import json

r = redis.Redis(host='localhost', port=6379, db=0)

def save_long_term(user_id: str, key: str, value: str):
    data = r.get(f"user:{user_id}") or "{}"
    memory = json.loads(data)
    memory[key] = value
    r.set(f"user:{user_id}", json.dumps(memory))

def load_long_term(user_id: str) -> dict:
    data = r.get(f"user:{user_id}")
    return json.loads(data) if data else {}
```

### 1.3 JSON Episodic Log

- **Mục đích:** Ghi lại các sự kiện/trải nghiệm đáng nhớ dưới dạng episodic memory.
- **Yêu cầu:**
  - Lưu mỗi episode dưới dạng JSON object với timestamp, nội dung, và tag.
  - Append vào file `episodes.json` hoặc database JSON.
  - Hỗ trợ tìm kiếm theo thời gian hoặc keyword.

```python
import json
from datetime import datetime

def log_episode(event: str, tags: list[str], filepath="episodes.json"):
    episode = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event,
        "tags": tags
    }
    try:
        with open(filepath, "r") as f:
            episodes = json.load(f)
    except FileNotFoundError:
        episodes = []
    episodes.append(episode)
    with open(filepath, "w") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)
```

### 1.4 Chroma (Semantic Memory)

- **Mục đích:** Lưu và truy xuất thông tin theo ngữ nghĩa (vector similarity search).
- **Yêu cầu:**
  - Dùng `chromadb` để tạo collection.
  - Embed các đoạn text bằng embedding model (ví dụ: `text-embedding-3-small` hoặc `sentence-transformers`).
  - Hỗ trợ truy vấn `top-k` kết quả gần nhất.

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.Client()
ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="YOUR_API_KEY",
    model_name="text-embedding-3-small"
)
collection = client.get_or_create_collection(
    name="semantic_memory",
    embedding_function=ef
)

def store_semantic(doc_id: str, text: str, metadata: dict):
    collection.upsert(documents=[text], ids=[doc_id], metadatas=[metadata])

def query_semantic(query: str, n_results: int = 3):
    return collection.query(query_texts=[query], n_results=n_results)
```

---

## Phần 2 – Build Memory Router

Xây dựng một **memory router** có khả năng tự động chọn loại memory phù hợp dựa trên **query intent**.

### 2.1 Phân loại Intent

| Intent | Memory được dùng | Ví dụ query |
|---|---|---|
| **User preference** | Long-term (Redis) | "Tôi thích phim gì?", "Màu sắc yêu thích của tôi?" |
| **Factual recall** | Semantic (Chroma) | "Thủ đô của Pháp là gì?", "Giải thích về GPT-4" |
| **Experience recall** | Episodic (JSON) | "Hôm qua chúng ta đã nói về gì?", "Lần trước tôi hỏi điều gì?" |
| **Recent context** | Short-term (Buffer) | Câu hỏi follow-up trong cùng phiên |

### 2.2 Implement Router

```python
from enum import Enum

class MemoryType(Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"

def classify_intent(query: str) -> MemoryType:
    """
    Dùng LLM hoặc rule-based để phân loại intent của query.
    """
    # Có thể dùng LLM để classify, hoặc keyword matching
    preference_keywords = ["thích", "yêu thích", "prefer", "favorite", "sở thích"]
    experience_keywords = ["hôm qua", "lần trước", "trước đây", "đã nói", "last time"]
    
    query_lower = query.lower()
    
    if any(kw in query_lower for kw in preference_keywords):
        return MemoryType.LONG_TERM
    elif any(kw in query_lower for kw in experience_keywords):
        return MemoryType.EPISODIC
    else:
        # Default: dùng semantic search để tìm kiếm factual
        return MemoryType.SEMANTIC

def route_memory(query: str, user_id: str) -> str:
    """
    Router chính: nhận query, trả về context từ memory phù hợp.
    """
    intent = classify_intent(query)
    
    if intent == MemoryType.LONG_TERM:
        data = load_long_term(user_id)
        return f"User preferences: {json.dumps(data, ensure_ascii=False)}"
    
    elif intent == MemoryType.EPISODIC:
        with open("episodes.json") as f:
            episodes = json.load(f)
        recent = episodes[-5:]  # 5 episodes gần nhất
        return json.dumps(recent, ensure_ascii=False, indent=2)
    
    elif intent == MemoryType.SEMANTIC:
        results = query_semantic(query, n_results=3)
        return "\n".join(results["documents"][0])
    
    else:  # SHORT_TERM
        return short_term.load_memory_variables({})["chat_history"]
```

> **Gợi ý nâng cao:** Dùng LLM để classify intent thay vì keyword matching để đạt độ chính xác cao hơn.

---

## Phần 3 – Context Window Management

Quản lý context window để tránh vượt quá giới hạn token, với **priority-based eviction** theo 4 cấp độ ưu tiên.

### 3.1 4-Level Priority Hierarchy

| Level | Loại nội dung | Mô tả | Evict khi nào |
|---|---|---|---|
| **P1 – Critical** | System prompt, user identity | Không bao giờ xóa | Không bao giờ |
| **P2 – High** | Context hiện tại (last 3 turns) | Giữ lại nếu có thể | Chỉ khi thực sự cần |
| **P3 – Medium** | Memory context từ router | Tóm tắt nếu quá dài | Sau P4 |
| **P4 – Low** | Lịch sử xa, episodic cũ | Xóa trước tiên | Đầu tiên |

### 3.2 Implement Auto-trim

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

class ContextWindowManager:
    def __init__(self, max_tokens: int = 8000, model: str = "gpt-4"):
        self.max_tokens = max_tokens
        self.model = model
        # Mỗi slot có priority và nội dung
        self.slots = {
            "system": {"priority": 1, "content": ""},
            "recent_turns": {"priority": 2, "content": ""},
            "memory_context": {"priority": 3, "content": ""},
            "old_history": {"priority": 4, "content": ""},
        }
    
    def total_tokens(self) -> int:
        total = 0
        for slot in self.slots.values():
            total += count_tokens(slot["content"], self.model)
        return total
    
    def auto_trim(self):
        """
        Tự động trim khi gần đến giới hạn.
        Evict theo thứ tự priority từ thấp đến cao (P4 trước).
        """
        # Sắp xếp slots theo priority giảm dần (P4 trước)
        sorted_slots = sorted(
            self.slots.items(),
            key=lambda x: x[1]["priority"],
            reverse=True
        )
        
        while self.total_tokens() > self.max_tokens * 0.9:  # Trim khi >= 90%
            for slot_name, slot in sorted_slots:
                if slot["priority"] == 1:
                    break  # Không bao giờ xóa P1
                if slot["content"]:
                    # Cắt bớt 30% nội dung slot này
                    words = slot["content"].split()
                    slot["content"] = " ".join(words[:int(len(words) * 0.7)])
                    break
            else:
                break  # Không còn gì để trim
    
    def build_context(self) -> str:
        """Ghép tất cả slots thành context string để đưa vào LLM."""
        self.auto_trim()
        parts = []
        for slot in sorted(self.slots.values(), key=lambda x: x["priority"]):
            if slot["content"]:
                parts.append(slot["content"])
        return "\n\n".join(parts)
```

---

## Phần 4 – Benchmark

So sánh hiệu năng của agent **có memory** và **không có memory** trên 10 multi-turn conversations.

### 4.1 Thiết kế Benchmark

- **Số lượng:** 10 conversations, mỗi conversation có ít nhất 5 turns.
- **Chủ đề đa dạng:** bao gồm câu hỏi về preference, factual, và experience recall.
- **2 điều kiện thử nghiệm:**
  - `Agent A` – không có memory (stateless)
  - `Agent B` – có đầy đủ memory system từ Lab này

### 4.2 Các Metrics Cần Đo

#### Response Relevance Score (0–1)
Đánh giá mức độ phù hợp của câu trả lời với context cuộc hội thoại.

```python
def evaluate_relevance(query: str, response: str, context: str) -> float:
    """
    Dùng LLM-as-judge để cho điểm relevance từ 0.0 đến 1.0.
    """
    prompt = f"""
    Query: {query}
    Context: {context}
    Response: {response}
    
    Rate how relevant the response is to the query given the context.
    Reply with ONLY a number between 0.0 and 1.0.
    """
    # Gọi LLM và parse kết quả
    score = float(call_llm(prompt).strip())
    return score
```

#### Context Utilization Score (0–1)
Đo mức độ agent tận dụng được thông tin từ memory.

```python
def evaluate_context_utilization(response: str, memory_context: str) -> float:
    """
    Kiểm tra response có sử dụng thông tin từ memory_context không.
    """
    prompt = f"""
    Memory context available: {memory_context}
    Agent response: {response}
    
    Rate how well the agent utilized the memory context (0.0 = ignored, 1.0 = fully used).
    Reply with ONLY a number between 0.0 and 1.0.
    """
    return float(call_llm(prompt).strip())
```

#### Token Efficiency
Đo số token trung bình dùng per turn và memory hit rate.

```python
def calculate_token_efficiency(conversations: list[dict]) -> dict:
    total_tokens = 0
    memory_hits = 0
    total_turns = 0
    
    for conv in conversations:
        for turn in conv["turns"]:
            total_tokens += turn["tokens_used"]
            if turn.get("memory_retrieved"):
                memory_hits += 1
            total_turns += 1
    
    return {
        "avg_tokens_per_turn": total_tokens / total_turns,
        "memory_hit_rate": memory_hits / total_turns,
        "total_tokens": total_tokens
    }
```

### 4.3 Định dạng Kết quả Benchmark

Kết quả cần trình bày trong file `benchmark_report.md` với nội dung sau:

```markdown
## Benchmark Results

| Metric | Agent (No Memory) | Agent (With Memory) | Improvement |
|---|---|---|---|
| Avg Response Relevance | 0.XX | 0.XX | +XX% |
| Avg Context Utilization | 0.XX | 0.XX | +XX% |
| Memory Hit Rate | N/A | 0.XX | – |
| Avg Tokens/Turn | XXX | XXX | –XX% |

### Memory Hit Rate Analysis
- Short-term hits: XX%
- Long-term (Redis) hits: XX%
- Episodic hits: XX%
- Semantic hits: XX%

### Token Budget Breakdown (per turn, avg)
- System prompt: XXX tokens (XX%)
- Memory context: XXX tokens (XX%)
- Recent turns: XXX tokens (XX%)
- Response: XXX tokens (XX%)
```

---

## 📁 Cấu trúc GitHub Repo

```
lab17-memory-agent/
├── README.md                  # Mô tả project và cách chạy
├── requirements.txt
├── .env.example               # Template cho API keys
│
├── memory/
│   ├── __init__.py
│   ├── short_term.py          # ConversationBufferMemory
│   ├── long_term.py           # Redis backend
│   ├── episodic.py            # JSON episodic log
│   └── semantic.py            # Chroma vector store
│
├── router/
│   ├── __init__.py
│   └── memory_router.py       # Intent classification + routing
│
├── context_manager/
│   ├── __init__.py
│   └── window_manager.py      # Auto-trim + priority eviction
│
├── agent/
│   ├── __init__.py
│   └── agent.py               # Main agent tích hợp tất cả
│
├── benchmark/
│   ├── conversations.json     # 10 test conversations
│   ├── run_benchmark.py       # Script chạy benchmark
│   └── benchmark_report.md   # Kết quả benchmark (bảng so sánh)
│
└── episodes.json              # Episodic memory log (được tạo khi chạy)
```

---

## ✅ Checklist Nộp Bài (Điểm Tối Đa)

- [ ] **Memory Backends** – Cả 4 loại hoạt động độc lập và có thể gọi được
- [ ] **Memory Router** – Phân loại đúng intent với ít nhất 3 loại (preference / factual / experience)
- [ ] **Context Manager** – Auto-trim khi đạt ngưỡng token, eviction đúng thứ tự priority
- [ ] **Benchmark** – Đủ 10 conversations, đo đủ 3 metrics (relevance, utilization, token efficiency)
- [ ] **Benchmark Report** – Có bảng so sánh, memory hit rate analysis, token budget breakdown
- [ ] **GitHub Repo** – Code sạch, README rõ ràng, có hướng dẫn cài đặt và chạy
- [ ] **Code Quality** – Có docstring, error handling, không hardcode API key trong code

---

## 💡Bonus

- Dùng **LLM-as-classifier** cho memory router thay vì keyword matching → độ chính xác cao hơn.
- Thêm **memory summarization**: tóm tắt episodic log cũ thay vì xóa hẳn.
- Implement **memory confidence score**: mỗi kết quả từ memory có điểm tin cậy, chỉ dùng khi score đủ cao.
- Visualize **context utilization** qua từng turn bằng biểu đồ.

---
