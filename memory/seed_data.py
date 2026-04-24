"""
Seed data for semantic memory — pre-loads FAQ/knowledge chunks into ChromaDB
so that semantic retrieval tests have meaningful content to retrieve.
"""

from memory.semantic import store_semantic

FAQ_CHUNKS = [
    {
        "id": "faq_refund_policy",
        "text": "Chính sách hoàn tiền của VinUniversity: Sinh viên có thể yêu cầu hoàn tiền học phí trong vòng 14 ngày kể từ ngày bắt đầu khóa học. Điều kiện hoàn tiền bao gồm: (1) Nộp đơn xin rút học phần, (2) Không có vi phạm kỷ luật, (3) Đã thanh toán đầy đủ học phí. Mức hoàn tiền: 100% nếu rút trong 7 ngày đầu, 50% nếu rút trong ngày 8-14, không hoàn tiền sau 14 ngày.",
        "metadata": {"category": "faq", "topic": "refund"}
    },
    {
        "id": "faq_scholarship",
        "text": "Học bổng VinUniversity: VinUniversity cung cấp nhiều loại học bổng dựa trên thành tích học tập và hoàn cảnh gia đình. Học bổng toàn phần bao gồm 100% học phí và hỗ trợ sinh hoạt phí. Sinh viên cần duy trì GPA tối thiểu 3.0/4.0 để giữ học bổng. Hạn nộp hồ sơ học bổng thường là tháng 3 hàng năm.",
        "metadata": {"category": "faq", "topic": "scholarship"}
    },
    {
        "id": "knowledge_docker",
        "text": "Docker là một nền tảng mã nguồn mở giúp tự động hóa việc triển khai ứng dụng bên trong các container phần mềm. Docker cho phép đóng gói ứng dụng cùng với tất cả các dependency vào một container nhẹ, di động. Docker Compose dùng để định nghĩa và chạy các ứng dụng multi-container. Khi dùng Docker Compose, các service kết nối với nhau bằng service name thay vì IP.",
        "metadata": {"category": "knowledge", "topic": "docker"}
    },
    {
        "id": "knowledge_container",
        "text": "Container là một đơn vị phần mềm chuẩn hóa, đóng gói code và tất cả dependency để ứng dụng chạy nhanh chóng và đáng tin cậy từ môi trường này sang môi trường khác. Container khác với Virtual Machine ở chỗ nó chia sẻ kernel của hệ điều hành host, do đó nhẹ và nhanh hơn nhiều.",
        "metadata": {"category": "knowledge", "topic": "container"}
    },
    {
        "id": "knowledge_microservices",
        "text": "Microservices (Kiến trúc vi dịch vụ) là một phương pháp kiến trúc phần mềm trong đó ứng dụng được chia thành các dịch vụ nhỏ, độc lập, mỗi dịch vụ đảm nhiệm một chức năng nghiệp vụ cụ thể. Các microservice giao tiếp qua API (thường là REST hoặc gRPC). Ưu điểm: dễ scale, deploy độc lập, team nhỏ quản lý. Nhược điểm: phức tạp về network, monitoring, data consistency.",
        "metadata": {"category": "knowledge", "topic": "microservices"}
    },
    {
        "id": "knowledge_vingroup",
        "text": "Tập đoàn Vingroup là một trong những tập đoàn kinh tế tư nhân lớn nhất Việt Nam, được thành lập bởi ông Phạm Nhật Vượng. Vingroup hoạt động trong nhiều lĩnh vực như bất động sản (Vinhomes), bán lẻ (VinMart), ô tô (VinFast), giáo dục (VinUniversity), y tế (Vinmec), và công nghệ (VinAI, VinBigData). Chủ tịch HĐQT hiện tại là ông Phạm Nhật Vượng.",
        "metadata": {"category": "knowledge", "topic": "vingroup"}
    },
    {
        "id": "knowledge_newyork",
        "text": "New York City có diện tích khoảng 783.8 km² (302.6 dặm vuông), bao gồm 5 quận (borough): Manhattan, Brooklyn, Queens, The Bronx, và Staten Island. Dân số khoảng 8.3 triệu người, là thành phố đông dân nhất nước Mỹ. Thủ đô của nước Mỹ là Washington D.C., không phải New York.",
        "metadata": {"category": "knowledge", "topic": "newyork"}
    },
    {
        "id": "knowledge_japan",
        "text": "Thủ đô của Nhật Bản là Tokyo, một trong những thành phố lớn nhất thế giới với diện tích khoảng 2,194 km² và dân số hơn 13 triệu người. Tokyo là trung tâm kinh tế, chính trị và văn hóa của Nhật Bản. Ngôn ngữ chính là tiếng Nhật (Japanese).",
        "metadata": {"category": "knowledge", "topic": "japan"}
    }
]


def seed_semantic_memory():
    """
    Pre-loads FAQ and knowledge chunks into ChromaDB for benchmark testing.
    """
    print("Seeding semantic memory with FAQ/knowledge chunks...")
    for chunk in FAQ_CHUNKS:
        store_semantic(chunk["id"], chunk["text"], chunk.get("metadata", {}))
    print(f"  Loaded {len(FAQ_CHUNKS)} knowledge chunks into ChromaDB.")


if __name__ == "__main__":
    seed_semantic_memory()
