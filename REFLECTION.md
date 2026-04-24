# Lab 17: Reflection on Privacy and Limitations

## 1. Memory nào giúp agent nhất?

- **Hữu ích nhất**: **Long-term profile (Redis/dict)** là memory giúp agent nhiều nhất. Nó cho phép agent "nhớ" thông tin cá nhân hóa của user (tên, sở thích, dị ứng, nghề nghiệp) và sử dụng chúng xuyên suốt các cuộc hội thoại. Không có long-term profile, agent sẽ hỏi lại tên user mỗi lần, tạo trải nghiệm rất kém.

- **Hữu ích thứ hai**: **Episodic memory** giúp agent nhớ các sự kiện quan trọng đã xảy ra (ví dụ: "user đã fix bug hệ thống", "user đã học xong Docker"), tạo cảm giác agent thực sự "đồng hành" cùng user.

- **Hữu ích cho kiến thức**: **Semantic memory (ChromaDB)** giúp agent truy xuất thông tin chính xác từ knowledge base thay vì hallucinate, đặc biệt với FAQ hoặc tài liệu chuyên ngành.

## 2. Memory nào rủi ro nhất nếu retrieve sai?

### Rủi ro PII (Personally Identifiable Information)

**Long-term profile** chứa rủi ro PII cao nhất vì nó lưu trữ trực tiếp:
- Tên thật của user
- Tình trạng sức khỏe (dị ứng: đậu nành, sữa bò)
- Nghề nghiệp, trường học
- Sở thích cá nhân

Nếu hệ thống **retrieve sai user_id** (ví dụ: lấy profile của User A ghép vào prompt của User B), đây là **rò rỉ dữ liệu chéo** nghiêm trọng. Ngay cả với cùng một user, việc nhầm lẫn context (ví dụ: nhầm dị ứng thuốc vs dị ứng thức ăn) có thể gây hậu quả nguy hiểm nếu agent đóng vai trò tư vấn y tế.

**Episodic memory** cũng chứa rủi ro khi ghi lại các sự kiện nhạy cảm mà user chia sẻ (ví dụ: "tôi vừa bị sa thải", "tôi đang bị bệnh").

### Ví dụ cụ thể rủi ro retrieval sai

```
Scenario: User A có profile {"allergy": "đậu nành"}
           User B có profile {"allergy": "sữa bò"}

Bug: System trả về profile User A khi User B hỏi "tôi dị ứng gì?"
Kết quả: Agent nói User B dị ứng đậu nành → User B uống sữa đậu nành → phản ứng dị ứng thật từ sữa bò
```

## 3. Quản lý vòng đời dữ liệu (Deletion, TTL, Consent)

### Xóa memory theo yêu cầu user

Nếu user yêu cầu "Hãy quên mọi thứ về tôi" (Right to be Forgotten — GDPR Article 17), hệ thống cần:

| Backend | Cách xóa | Độ khó |
|---------|---------|--------|
| Long-term (Redis) | `r.delete(f"user:{user_id}")` | Dễ — xóa 1 key |
| Episodic (JSON) | Filter và xóa tất cả entries có `user_id` match | Trung bình — cần parse file |
| Semantic (ChromaDB) | `collection.delete(where={"user_id": user_id})` | Trung bình — cần metadata filtering |
| Short-term (RAM) | `self.chat_history.clear()` | Dễ — xóa list |

### TTL (Time-To-Live)

Hiện tại hệ thống **không có TTL** — dữ liệu lưu mãi mãi. Cần cải thiện:
- **Redis TTL**: `r.setex(f"user:{user_id}", 30*86400, data)` — auto-expire sau 30 ngày không tương tác
- **Episodic TTL**: Chỉ giữ episodes trong 90 ngày gần nhất, tự động xóa entries cũ hơn
- **Semantic**: Ít cần TTL vì thường chứa kiến thức chung, không phải dữ liệu cá nhân

### User Consent

Hệ thống hiện tại **thiếu cơ chế consent**. Cần bổ sung:
1. **Thông báo lần đầu**: "Tôi sẽ ghi nhớ thông tin bạn chia sẻ để cá nhân hóa trải nghiệm. Bạn có đồng ý không?"
2. **Opt-out**: Cho phép user tắt memory logging bất cứ lúc nào
3. **Transparency**: Cho phép user xem lại tất cả facts đã lưu về họ (`/show-memory`)

## 4. Hạn chế kỹ thuật của giải pháp hiện tại

### 4.1. Scalability của Episodic Memory
Hiện tại episodic memory sử dụng file `episodes.json` duy nhất, **không phân biệt user_id**. Mọi user đều ghi vào cùng một file. Khi scale lên nhiều user:
- File JSON sẽ rất lớn → đọc/ghi chậm (I/O bottleneck)
- Không thể query theo user → trả về episodes của user khác
- **Giải pháp**: Chuyển sang MongoDB/PostgreSQL hoặc vector DB có metadata filtering theo user_id

### 4.2. Chi phí LLM Extraction
Cơ chế `extract_memory_node` gọi LLM **trước mỗi response** để extract facts/episodes. Điều này:
- Tăng gấp đôi latency (2 LLM calls per turn thay vì 1)
- Tăng gấp đôi chi phí token
- **Giải pháp**: Chỉ extract khi confidence score cao, hoặc dùng small model riêng cho extraction

### 4.3. Context Window Limits
Dù có `ContextWindowManager` với auto-trim, nếu profile có hàng trăm facts → inject toàn bộ vào prompt sẽ chiếm hết token budget, làm giảm chất lượng response.
- **Giải pháp**: Relevance-based profile injection — chỉ inject facts liên quan đến câu hỏi hiện tại

### 4.4. Embedding Quality
ChromaDB sử dụng `all-MiniLM-L6-v2` — model embedding nhỏ, chất lượng trung bình. Đặc biệt kém với tiếng Việt vì model được train chủ yếu trên tiếng Anh.
- **Giải pháp**: Sử dụng multilingual embedding model (ví dụ: `multilingual-e5-large`) hoặc fine-tune trên corpus tiếng Việt

### 4.5. Conflict Detection hạn chế
LLM-based conflict detection phụ thuộc vào prompt engineering. Nếu user sửa fact một cách gián tiếp (ví dụ: "hôm nay tôi chuyển sang ăn chay" ngầm thay đổi dietary preferences), LLM có thể không nhận ra đây là conflict.
- **Giải pháp**: Xây dựng explicit conflict detection rules + LLM fallback

### 4.6. Multi-user Isolation
Hiện tại `episodes.json` không có user_id filtering — tất cả users chia sẻ cùng episodic memory. Đây là lỗi bảo mật nghiêm trọng trong production.
- **Giải pháp**: Thêm `user_id` vào mỗi episode entry và filter khi retrieve
