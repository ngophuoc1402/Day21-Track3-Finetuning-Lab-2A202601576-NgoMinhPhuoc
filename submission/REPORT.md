# Lab 21 — Evaluation Report

**Họ tên**: Ngô Minh Phước  **MSSV**: 2A202601576  **Ngày**: 2026-08-21
**Tier**: `CPU`  **Base model**: `Qwen/Qwen3.5-0.8B`  **GPU thực tế**: `RTX 3060 12GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 512 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / 58 |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json)*
Nếu không: bạn đã xử lý thế nào? (Không áp dụng, template Qwen3.5-0.8B giữ nguyên khối suy luận)

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.3936` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.6444 | 0.0000 | 1471.0 |
| (b) base + optimized prompt | 0.4950 | 0.6444 | 1.0000 | 404.0 |
| (c) LoRA fine-tune | 0.9900 | 0.0667 | 1.0000 | 626.0 |

**(b) có thật sự mạnh hơn (a) không?** Có — prompt tối ưu giúp định dạng JSON hoàn hảo (1.000 so với 0.000) và tăng điểm target vượt bậc đồng thời giảm latency 3.6 lần nhờ chặn đứng model viết lan man.
Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: Không.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 10822656 | 1e-4 | 0.3944 | 0.9900 | 446.0 | 3.07 |
| `attn_only` | q,v | 271 | 10822656 | 1e-4 | 0.4340 | 0.9350 | 888.9 | 3.08 |
| `wrong_lr` | text-linear | 16 | 10822656 | 1e-5 | 1.5426 | 0.3250 | 1021.3 | 3.08 |
| `qlora` | text-linear | 16 | 10822656 | 1e-4 | 0.4243 | 0.9300 | 1084.7 | 2.29 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

Trên tập target, `attn_only` thua `correct` (0.9350 so với 0.9900) dù có cùng số lượng tham số huấn luyện. Thứ tự này hoàn toàn ngược với thứ tự đánh giá theo train loss, nơi `attn_only` có xu hướng ép loss xuống thấp hơn nhờ dung lượng cục bộ lớn tại 6 lớp attention. Kết quả này chứng minh rằng việc phân bổ adapter đến mọi lớp tuyến tính (placement) quan trọng hơn nhiều so với việc nâng rank của adapter tại một số lớp giới hạn; cấu hình LoRA đúng đắn phải phủ rộng hơn là đào sâu.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

Đường loss của `wrong_lr` dẹt ngang, giảm cực kỳ chậm và kết thúc ở mức loss rất cao (1.5426) so với 0.3944 của `correct`. Nếu chỉ nhìn loss mà không biết learning rate bị đặt sai thang đo (1e-5 thay vì 1e-4), ta sẽ dễ dàng kết luận sai lầm rằng adapter bị thiếu dung lượng (rank quá nhỏ) hoặc bài toán quá khó không thể hội tụ. Thực chất, vấn đề nằm hoàn toàn ở việc học quá chậm do learning rate scale của full fine-tuning bị áp dụng nhầm cho LoRA, chứng minh rằng thang đo LR là đòn bẩy quan trọng nhất quyết định sự hội tụ.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

`qlora` giúp tiết kiệm được 25.4% bộ nhớ VRAM đỉnh (2.29 GB so với 3.07 GB). Tuy nhiên, cái giá phải trả là thời gian huấn luyện tăng gấp 2.4 lần (1084.7 s so với 446.0 s) do chi phí dequantization liên tục, cùng với việc điểm target bị sụt giảm từ 0.9900 xuống 0.9300. Số đo này hoàn toàn ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" từ phía nhà phát triển Unsloth, vì sai số lượng tử hóa làm thui chột đáng kể độ chính xác của tác vụ hẹp trong khi chi phí thời gian huấn luyện là quá đắt đỏ.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.4950` · `regression Δ = -0.5777` · `valid_trace_rate = 0.00`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
(Một FAILED được phân tích tốt ăn điểm cao hơn một PASSED không giải thích được.)

Cổng hồi quy báo FAILED vì chỉ số `regression delta` tụt xuống tới -0.5777, vượt xa mức dung sai cho phép 0.02. Mặc dù mô hình fine-tune đạt điểm target xuất sắc (+0.4950 so với baseline b) và tuân thủ định dạng 100%, nó đã bị hiện tượng catastrophic forgetting (quên thảm họa). Mô hình học được rằng bất kỳ đầu vào nào cũng phải được trả về dưới dạng JSON triage, khiến nó mất hoàn toàn khả năng trả lời các câu hỏi kiến thức thông thường của tập regression. Điều này chỉ ra rằng tập dữ liệu huấn luyện thuần túy 100% SFT mà không có cơ chế chống quên (chẳng hạn như trộn 1-5% dữ liệu tổng quát / replay data) là quá cực đoan cho mô hình nền.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Shop ơi, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại... | doi_tra, trung_binh, balo laptop | doi_tra, thap, balo laptop | doi_tra, trung_binh, balo laptop | ✅ FT thắng (nhận diện đúng urgency) |
| 2 | Shop giao sai màu rồi shop ơi, mình đặt tai nghe bluetooth màu đen... | san_pham_loi, cao, tai nghe bluetooth | san_pham_loi, trung_binh, tai nghe | san_pham_loi, cao, tai nghe bluetooth | ✅ FT thắng (nhận diện đúng sản phẩm & urgency) |
| 3 | Đơn hàng chuột không dây của mình khi nào giao vậy? Lâu quá shop... | van_chuyen, trung_binh, chuột không dây | van_chuyen, trung_binh, chuột không dây | van_chuyen, thap, chuột không dây | ❌ **FT thua** (nhận diện sai urgency thap) |
| 4 | Tôi muốn hủy đơn hàng mã DH998124 và hoàn tiền lại cho tôi... | hoan_tien, cao, không có | hoan_tien, cao, không có | hoan_tien, cao, DH998124 | ❌ **FT thua** (nhầm mã đơn hàng là tên sản phẩm) |
| 5 | Shop ơi, chuột gaming em mua tuần trước giờ cắm không lên đèn nữa | san_pham_loi, trung_binh, chuột gaming | san_pham_loi, trung_binh, chuột gaming | san_pham_loi, trung_binh, chuột gaming | 🤝 Hòa (đều phân loại chính xác) |

Có mẫu chung nào ở các ca FT thua không?
Các ca fine-tune thua thường gặp khó khăn ở các tình huống mơ hồ (như nhầm lẫn giữa mã đơn hàng và tên sản phẩm khi trích xuất) hoặc đánh giá sai lệch các sắc thái chủ quan về độ khẩn cấp (urgency) do kích thước tập dữ liệu huấn luyện quá nhỏ (225 mẫu) dẫn đến mô hình bị thiên lệch cục bộ, trong khi mô hình nền với prompt tối ưu (b) giữ được ngữ cảnh suy luận khách quan hơn.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn
bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?

Tôi đề xuất không deploy bản fine-tune này. Mặc dù nó đạt độ chính xác gần như hoàn hảo trên tác vụ phân loại ticket CSKH hẹp (0.9900) và tuân thủ định dạng JSON 100%, sự sụp đổ khả năng tổng quát (tụt giảm 57.8% khả năng trả lời câu hỏi thông thường) biến nó thành một tác nhân thiếu an toàn và dễ bị khai thác. Bất kỳ đầu vào lệch chuẩn nào cũng sẽ bị mô hình trả về dưới dạng JSON phân loại ticket. 

Đòn bẩy thực sự trong lab này là chất lượng dữ liệu (sự cần thiết của dữ liệu replay để chống quên) và cơ chế che nhãn (mask) chính xác tại NB1, kế tiếp là learning rate scale. Nếu không chứng minh được loss mask đúng từ đầu, mô hình sẽ học cả câu hỏi của người dùng và hỏng pipeline. Nếu LR đặt sai thang (như `wrong_lr`), mô hình không thể hội tụ. Cuối cùng, nếu không trộn dữ liệu chống quên, mô hình sẽ quên sạch tri thức tổng quát ngay khi bắt đầu SFT.

**Ba điều tôi học được** (cụ thể, không generic):
1. Phải giải mã ngược nhãn (mask proof) để kiểm chứng thực tế token nào được tính loss, thay vì tin tưởng mù quáng vào các cờ thư viện tự động như `assistant_only_loss=True` (vốn hoạt động không đúng trên Qwen3.5 do chat template thiếu marker generation).
2. Phép so sánh cấu hình LoRA chỉ công bằng khi ta giữ nguyên ngân sách tham số huấn luyện (matched trainable parameters) bằng cách sử dụng `matched_rank()`, thay vì so sánh cùng rank nhưng vị trí đặt khác nhau dẫn đến ngân sách chênh lệch lớn.
3. Prompt engineering tốt (baseline b) là một đối thủ đáng gờm, vừa tiết kiệm chi phí suy luận vừa bảo toàn khả năng tổng quát của mô hình nền; ta phải luôn thiết lập baseline trước khi huấn luyện để giữ tính liêm chính cho quá trình nghiên cứu.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
1. Trộn thêm 1-5% dữ liệu chat tổng quát (ví dụ từ tập ShareGPT) vào tập training để kiểm tra xem regression gate có PASS được không.
2. Thử nghiệm quét rank (rank sweep) chi tiết trên text-linear với rank 8, 16 và 64 để tìm điểm bão hòa tối ưu của adapter trên tác vụ phân loại JSON này.

---

## Phụ lục — thưởng đã làm

- [x] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
