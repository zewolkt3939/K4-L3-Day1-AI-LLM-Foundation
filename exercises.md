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

> Khi tăng temperature từ 0.0 lên 1.5, câu trả lời chuyển dần từ an toàn, lặp lại sang phong phú và ngẫu nhiên hơn. Ở mức 0.0 và 0.5, model đều chọn sự thật quen thuộc về Hang Sơn Đoòng với câu từ khá cố định. Khi lên 1.0 và 1.5, văn phong linh hoạt hơn hẳn và model đổi sang chủ đề khác (Vịnh Hạ Long), cho thấy temperature càng cao thì phản hồi càng đa dạng nhưng cũng dễ lan man hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**

> Mình sẽ chọn khoảng **0.0 đến 0.2**. Chatbot hỗ trợ khách hàng cần ưu tiên tính chính xác và nhất quán theo chính sách công ty (tránh trường hợp hai khách hỏi cùng một vấn đề mà nhận câu trả lời khác nhau). Mức nhiệt độ thấp giúp giảm thiểu nguy cơ model bịa thông tin (hallucination), trong khi câu từ vẫn đủ tự nhiên và thân thiện.

### Câu 1.3 — Đánh đổi chi phí

Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

> - **Chênh lệch chi phí:** Mỗi ngày hệ thống tốn $10.000 \times 3 \times 350 = 10.500.000$ output tokens ($10.500\text{K tokens}$). Dùng GPT-4o tốn $\$105$/ngày, trong khi GPT-4o-mini chỉ tốn $\$6.3$/ngày. Như vậy, GPT-4o đắt hơn mini khoảng **16.7 lần** ($0.010 / 0.0006$).
> - **Khi nào nên dùng GPT-4o:** Các tác vụ cần suy luận phức tạp và độ chính xác cao như rà soát hợp đồng pháp lý, tư vấn tài chính hay chẩn đoán kỹ thuật.
> - **Khi nào nên dùng GPT-4o-mini:** Các tác vụ đơn giản, lặp lại nhiều như phân loại ý định người dùng (intent classification), tóm tắt tin nhắn ngắn, hoặc trả lời FAQ có sẵn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona

Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:

- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)

> Bản cho học sinh tiểu học viết rất ngắn gọn, dùng ví dụ cuốn sổ ghi điểm khi chơi trò chơi và từ ngữ đơn giản để trẻ dễ hiểu. Ngược lại, bản cho chuyên gia tài chính dùng nhiều thuật ngữ chuyên ngành như "sổ cái phân tán", "tính bất biến" và giải thích cơ chế sâu hơn. Qua đó thấy system prompt quyết định trực tiếp tông giọng, vốn từ và độ sâu kỹ thuật của câu trả lời. Nhờ đặt đúng persona, cùng một mô hình có thể linh hoạt điều chỉnh cách giải thích cho từng đối tượng người nghe khác nhau.

### Câu 2.2 — tiktoken vs đếm từ

Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**

> - **So sánh số liệu:** Với đoạn văn thử nghiệm 117 từ, tiktoken (`o200k_base`) đếm được 153 tokens, còn công thức ước lượng `117 / 0.75` ra 156 tokens, chênh lệch thực tế chỉ khoảng **1.9%**.
> - **Vì sao tiếng Việt tốn token hơn tiếng Anh:**
>   1. Bộ tokenizer (BPE) được train chủ yếu bằng tiếng Anh, nên đa số từ tiếng Anh phổ biến nằm trọn trong 1 token.
>   2. Tiếng Việt có nhiều nguyên âm có dấu thanh và ký tự Unicode đa byte, nên tokenizer hay phải chẻ một âm tiết thành 2–3 subword tokens.
>   3. Từ ghép tiếng Việt có khoảng trắng giữa các âm tiết, nên model tokenize từng tiếng một chứ không gộp cả từ lại được.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming

**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)

> Streaming quan trọng nhất khi xây dựng giao diện tương tác trực tiếp với người dùng (chatbot web, terminal, trợ lý ảo). Việc trả về từng token liên tục giúp tối ưu thời gian phản hồi đầu tiên (TTFT), người dùng có thể đọc ngay lập tức thay vì phải sốt ruột chờ 5–10 giây trước màn hình trống. Ngược lại, non-streaming lại phù hợp hơn khi chạy tác vụ ngầm theo lô (batch job), khi cần model trả về dữ liệu có cấu trúc (JSON, SQL) để code parse nguyên khối, hoặc khi cần quét kiểm duyệt an toàn (guardrails) toàn bộ nội dung trước khi hiển thị cho người dùng.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?

**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**

> - **Lợi thế của Exponential Backoff:** Khi gặp lỗi quá tải (429 hoặc 503), việc nhân đôi thời gian chờ sau mỗi lần thử ($0.1s \rightarrow 0.2s \rightarrow 0.4s \dots$) giúp kéo giãn tần suất gọi API, cho server thời gian xử lý hết hàng đợi đang nghẽn và tự phục hồi.
> - **Nếu hàng nghìn client retry cố định (ví dụ 1 giây):** Sẽ gây ra hiện tượng bão yêu cầu (_thundering herd_). Cứ đúng 1 giây sau, tất cả client lại đồng loạt dội request về cùng một thời điểm, khiến server vừa gượng dậy đã bị nghẽn tải sập tiếp. Do đó, exponential backoff (thường kèm một chút ngẫu nhiên jitter) giúp rải đều tải ra và bảo vệ hệ thống ổn định hơn.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona

**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**

> - **System prompt đã chọn:**
>   `"Bạn là trợ giảng thân thiện của khóa AI, chuyên về Python và LLM. Hãy trả lời ngắn gọn, dùng gạch đầu dòng và giải thích bằng tiếng Việt dễ hiểu."`
> - **Lý do lựa chọn từ ngữ:**
>   1. _"Trả lời ngắn gọn, dùng gạch đầu dòng":_ Vì trợ lý chạy trên terminal có không gian hẹp, câu trả lời ngắn gọn và chia ý rõ ràng sẽ giúp học viên dễ đọc, không phải cuộn màn hình nhiều và cũng tiết kiệm đáng kể token đầu ra.
>   2. _"Giải thích bằng tiếng Việt dễ hiểu":_ Giúp người học tiếp thu kiến thức tự nhiên, đồng thời ngăn model tự động chuyển sang tiếng Anh khi gặp các thuật ngữ kỹ thuật phức tạp.

### Câu 4.2 — Hạn chế & cải thiện

**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**

> - **Hạn chế lớn nhất:** Lịch sử trò chuyện chỉ giữ lại 3 lượt gần nhất (`history[-6:]`). Nếu nói chuyện dài hơn 3 câu, bot sẽ quên sạch các thông tin ban đầu (như đề bài, biến đã khai báo hay mục tiêu người dùng), dẫn đến câu trả lời bị cụt ý hoặc mâu thuẫn với phía trước.
> - **Đề xuất cải thiện: Kết hợp tóm tắt hội thoại với bộ đệm (Summary Buffer Memory).**
>   - _Cách triển khai:_ Giữ lại 2 lượt chat mới nhất trong bộ nhớ; khi các tin nhắn cũ hơn bị đẩy ra ngoài, kích hoạt một lệnh gọi ngầm dùng model nhỏ (`gpt-4o-mini`) để tóm tắt chúng thành 1 đoạn ngắn lưu vào biến `summary`. Khi gọi API, gửi kèm: `[System Prompt] + [Đoạn tóm tắt cũ] + [2 lượt chat gần nhất] + [Câu hỏi mới]`. Cách này vừa giữ được mạch hội thoại xuyên suốt mà không sợ bị phình số lượng token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
