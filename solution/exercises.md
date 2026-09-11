# K4 — Ngày 1: Bài Tập & Phản Ánh

## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature

Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)

> Qua bốn phản hồi thực tế, ta thấy rõ quy luật: Ở mức 0.0, mô hình mang tính tiền định (deterministic) cao nhất, trả về thông tin phổ biến nhất (Hang Sơn Đoòng) với câu từ khuôn mẫu và lặp lại ổn định; khi tăng lên 0.5 và 1.0, cách diễn đạt bắt đầu linh hoạt, tự nhiên hơn và bổ sung chi tiết đa dạng (thời gian hình thành hang động 2-5 triệu năm trước); đến mức 1.5, xác suất lấy mẫu phân tán mạnh khiến mô hình chuyển sang chọn địa danh khác (Vịnh Hạ Long) với văn phong tự do hơn. Quy luật tổng quát là temperature tỷ lệ thuận với tính ngẫu nhiên và đa dạng của từ vựng: temperature thấp giúp câu trả lời nhất quán, chuẩn xác; temperature cao tăng tính sáng tạo nhưng dễ dẫn đến sai lệch ngữ nghĩa hoặc hallucination.

### Câu 1.2 — Chọn temperature cho sản phẩm

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**

> Tôi sẽ đặt temperature trong khoảng **0.1 đến 0.3** (hoặc 0.0 khi tra cứu chính sách/FAQ). Chatbot hỗ trợ khách hàng đòi hỏi tính chính xác, trung thực và bám sát chính sách sản phẩm/điều khoản bảo hành của doanh nghiệp; nhiệt độ thấp giúp giảm thiểu tối đa hiện tượng "bịa đặt" (hallucination) và đảm bảo tính nhất quán cao (hai khách hàng hỏi cùng một chính sách hoàn tiền sẽ nhận được câu trả lời đồng nhất, tránh gây tranh chấp). Mức 0.1–0.2 vẫn giữ cho câu từ tự nhiên mà không phương hại đến tính chính xác.

### Câu 1.3 — Đánh đổi chi phí

Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

> - **Tính toán chi phí:** Tổng lượt gọi mỗi ngày là $10.000 \times 3 = 30.000$ requests, tương ứng $30.000 \times 350 = 10.500.000$ output tokens ($10.500$ nghìn tokens). Với giá GPT-4o ($0.010/1K), chi phí là $\$105.00$/ngày ($\sim \$3.150$/tháng). Với giá GPT-4o-mini ($0.0006/1K), chi phí chỉ là $\$6.30$/ngày ($\sim \$189$/tháng). Do đó, GPT-4o đắt hơn GPT-4o-mini đúng **16.67 lần** ($0.010 / 0.0006$).
> - **Trường hợp GPT-4o xứng đáng:** Tác vụ tư vấn pháp lý, thẩm định hợp đồng tài chính phức tạp hoặc chẩn đoán kỹ thuật đòi hỏi suy luận logic nhiều bước (complex reasoning), nơi một sai sót nhỏ có thể gây thiệt hại tài chính nghiêm trọng.
> - **Trường hợp nên dùng GPT-4o-mini:** Tác vụ phân loại ý định người dùng (intent classification), tóm tắt tin nhắn ngắn, hoặc trả lời các câu hỏi thường gặp (FAQ) có lưu lượng truy cập lớn cần phản hồi tức thì với chi phí tối thiểu.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona

Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:

- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)

> Hai phản hồi có sự khác biệt rõ rệt: Persona giáo viên tiểu học dùng hình ảnh ẩn dụ cuốn sổ ghi chép điểm số khi chơi trò chơi, mỗi trang là một khối, từ ngữ mộc mạc và câu văn ngắn giúp trẻ 8 tuổi dễ hình dung; trong khi persona chuyên gia tài chính dùng thuật ngữ học thuật chuẩn mực như "công nghệ sổ cái phân tán (DLT)", "tính bất biến", "mã hóa liên kết theo thứ tự thời gian". System prompt hoạt động như một chỉ thị đạo diễn cấp cao định hình toàn bộ phong cách hành văn, mức độ sâu sắc kỹ thuật và lựa chọn từ vựng của mô hình. Điều này chứng minh rằng cùng một tri thức nền tảng, system prompt có khả năng biến đổi linh hoạt cách truyền đạt để nhắm đúng đối tượng mục tiêu.

### Câu 2.2 — tiktoken vs đếm từ

Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**

> - **So sánh số liệu:** Thử nghiệm trên đoạn văn mẫu tiếng Việt 117 từ, `count_tokens` (tiktoken o200k_base) trả về 153 tokens, trong khi công thức ước lượng `117 / 0.75` ra 156 tokens; mức chênh lệch thực tế là khoảng **1.92%**.
> - **Vì sao tiếng Việt tốn nhiều token hơn tiếng Anh:** Thứ nhất, tokenizer (như BPE) được huấn luyện chủ yếu trên văn bản tiếng Anh, nên hầu hết từ vựng tiếng Anh thông dụng đều nằm trọn trong 1 token duy nhất. Thứ hai, tiếng Việt sử dụng nhiều nguyên âm có dấu thanh (như ư, ơ, ê, dấu hỏi, ngã, nặng) được mã hóa bằng nhiều byte UTF-8, khiến tokenizer thường phải tách mỗi âm tiết thành 2–3 subword tokens. Thứ ba, từ ghép tiếng Việt có khoảng trắng giữa các tiếng, khiến mô hình xử lý từng tiếng riêng lẻ thay vì gom thành một khối từ vựng hoàn chỉnh.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming

**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)

> Streaming quan trọng nhất trong các ứng dụng đối thoại tương tác thời gian thực giữa người và máy (như chatbot chăm sóc khách hàng, trợ lý lập trình CLI, voice assistant) vì nó tối ưu hóa triệt để chỉ số thời gian phản hồi đầu tiên (Time-to-First-Token - TTFT); thay vì bắt người dùng phải chờ đợi thụ động suốt 5–15 giây trước một màn hình trống, các token được in ra liên tục tạo cảm giác phản hồi tức thì và cho phép người dùng bắt đầu đọc thông tin ngay lập tức. Ngược lại, non-streaming lại phù hợp hơn trong các trường hợp: (1) Các tác vụ xử lý hàng loạt chạy ngầm (batch processing/offline ETL) không có sự tương tác trực tiếp của con người; (2) Khi hệ thống cần sinh dữ liệu có cấu trúc định dạng nghiêm ngặt (như JSON, XML, SQL) cần nhận đầy đủ toàn bộ payload trước khi đưa vào hàm parse kiểm tra cú pháp; hoặc (3) Khi luồng xử lý bắt buộc phải chạy qua một bộ lọc kiểm duyệt an toàn (moderation/guardrails) để quét nội dung độc hại trước khi hiển thị cho người dùng cuối.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?

**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**

> - **Lợi thế của Exponential Backoff:** Khi API gặp sự cố quá tải hoặc nghẽn mạng tạm thời (HTTP 429 hoặc 503), chiến lược tăng thời gian chờ gấp đôi sau mỗi lần thử lại ($0.1s \rightarrow 0.2s \rightarrow 0.4s \dots$) giúp kéo giãn khoảng cách giữa các lần gửi request, tạo ra "khoảng thở" cần thiết để hệ thống backend kịp xử lý hàng đợi đang ùn ứ và giải phóng tài nguyên.
> - **Nếu hàng nghìn client retry với delay cố định:** Sẽ lập tức kích hoạt hiện tượng "bão đồng bộ" hay **Thundering Herd Problem**. Khi server vừa phục hồi sau đúng 1 giây, hàng nghìn client đồng thời dội lại các request cùng một thời điểm chính xác, gây ra một đợt sốc tải mới khiến server sập trở lại ngay lập tức (cascading failure). Do đó, exponential backoff (thường kèm một lượng nhiễu ngẫu nhiên jitter) là tiêu chuẩn bắt buộc để bảo vệ sự ổn định của cả client lẫn server.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona

**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**

> - **Persona được lựa chọn:**
>   `"Bạn là trợ giảng AI thông minh và tận tâm của khóa học AI Practical Competency (K4), chuyên gia về Python và mô hình ngôn ngữ lớn (LLM). Hãy trả lời ngắn gọn, có cấu trúc rõ ràng, dùng ví dụ trực quan và luôn giải thích bằng tiếng Việt chuẩn mực."`
> - **Giải thích các lựa chọn từ ngữ quan trọng:**
>   1. *"Trả lời ngắn gọn, có cấu trúc rõ ràng":* Do môi trường hoạt động là giao diện dòng lệnh (CLI/Terminal) có không gian hiển thị giới hạn, câu trả lời ngắn gọn, dùng gạch đầu dòng giúp người học tiếp thu thông tin nhanh chóng mà không cần cuộn trang nhiều lần, đồng thời tiết kiệm đáng kể lượng output token (giảm độ trễ và chi phí).
>   2. *"Luôn giải thích bằng tiếng Việt chuẩn mực":* Đảm bảo trải nghiệm thân thiện và dễ hiểu đối với học viên Việt Nam, ngăn mô hình tự động chuyển ngữ sang tiếng Anh khi gặp các thuật ngữ lập trình chuyên sâu, nhưng vẫn giữ nguyên các từ khóa kỹ thuật cần thiết trong ngoặc đơn để đối chiếu.

### Câu 4.2 — Hạn chế & cải thiện

**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**

> - **Hạn chế lớn nhất:** Quản lý ngữ cảnh chỉ dựa vào cửa sổ trượt 3 lượt gần nhất (`history = history[-6:]`). Khi phiên hội thoại kéo dài vượt quá 3 lượt hỏi-đáp, toàn bộ thông tin quan trọng ban đầu (như yêu cầu đề bài, các biến cấu hình, mục tiêu người dùng đã nêu) sẽ bị xóa bỏ hoàn toàn, khiến trợ lý bị "mất trí nhớ", dẫn đến câu trả lời thiếu mạch lạc hoặc mâu thuẫn với các lượt trước.
> - **Cải thiện cụ thể: Triển khai Conversation Summary Buffer Memory (Bộ nhớ tóm tắt kết hợp bộ đệm).**
>   - *Cách triển khai:* Duy trì một bộ đệm chứa 2 lượt hội thoại gần nhất và một biến chuỗi tóm tắt `conversation_summary`. Khi một lượt chat cũ chuẩn bị trôi ra khỏi bộ đệm, hệ thống sẽ kích hoạt một lời gọi API nền (sử dụng model nhẹ `gpt-4o-mini`) với nhiệm vụ tóm tắt thông tin quan trọng của các lượt cũ và cập nhật vào `conversation_summary`. Mỗi khi gửi request mới, cấu trúc messages sẽ gồm: `[System Persona] + [{"role": "system", "content": f"Tóm tắt ngữ cảnh cuộc hội thoại trước: {conversation_summary}"}] + [2 lượt gần nhất trong buffer] + [User message mới]`. Cơ chế này giữ được toàn bộ diễn biến cuộc đối thoại lâu dài mà lượng token gửi đi vẫn luôn nhỏ gọn và ổn định.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
