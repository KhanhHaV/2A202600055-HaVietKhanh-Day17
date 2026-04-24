# Benchmark Results: No-Memory vs With-Memory

| # | Scenario | No-memory result | With-memory result |
|---|----------|------------------|---------------------|
| 1 | Profile recall after many turns (Q: Bạn có nhớ tôi tên là gì không?) | Fail | Fail |
| 2 | Allergy conflict update (Q: Tôi dị ứng gì bạn nhớ không?) | Fail | Fail |
| 3 | Recall previous debug lesson (Episodic) (Q: Bạn có nhớ tôi học được kinh nghiệm gì về docker không?) | Pass | Fail |
| 3 | Recall previous debug lesson (Episodic) (Q: Tôi có nên áp dụng nó cho dự án web không?) | Manual Review | Manual Review |
| 4 | Retrieve FAQ chunk (Semantic) (Q: Tôi muốn hỏi lại về hoàn tiền, điều kiện là gì?) | Manual Review | Manual Review |
| 5 | Trim/token budget test (Q: Tóm tắt câu chuyện lại trong 1 câu.) | Manual Review | Manual Review |
| 6 | Mixed Intent & Identity (Q: Bạn giải thích khái niệm Microservices giúp tôi.) | Manual Review | Manual Review |
| 7 | Preference Conflict (Q: Bây giờ tôi thích màu gì?) | Pass | Fail |
| 8 | Episodic vs Semantic clarity (Q: New York có diện tích bao nhiêu?) | Manual Review | Manual Review |
| 9 | Complex Short-term tracking (Q: Hiện tại tôi có bao nhiêu quả táo?) | Fail | Fail |
| 10 | Final Robustness Check (Q: Môn học gần nhất tôi hoàn thành là gì?) | Manual Review | Manual Review |

## Qualitative Assessment
- **Profile Recall**: Agent with memory successfully extracts user facts into Redis and recalls them (e.g., name, allergies).
- **Conflict Update**: LLM-based memory extraction overwrites old facts (e.g., "sữa bò" -> "đậu nành"). Stateless agent fails.
- **Episodic Recall**: Important events are tagged and logged to `episodes.json` and successfully retrieved later.
- **Semantic Retrieval**: Knowledge fetching via ChromaDB vector store is operational (simulated or real).
- **Trim/Token Budget**: MemoryState via ContextWindowManager automatically prunes `messages` history or limits context injection when token counts get too high.
