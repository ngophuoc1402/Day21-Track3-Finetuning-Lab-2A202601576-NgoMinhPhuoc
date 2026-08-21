# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Tôi rất ngạc nhiên khi thấy adapter fine-tune ban đầu có thể đạt target=0.0000 chỉ vì sự lệch pha (mismatch) giữa cấu trúc prompt lúc huấn luyện và lúc đánh giá, mặc dù đồ thị training loss trông rất đẹp. Điều này chỉ ra rằng các thư viện tự động có thể che giấu những lỗi logic nghiêm trọng bên dưới.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Tôi mất nhiều thời gian nhất để kiểm tra và căn chỉnh loss mask cùng chat template nhằm đảm bảo phần loss chỉ tính trên câu trả lời của assistant. Đây không phải chỗ tôi dự đoán ban đầu (tôi nghĩ việc chạy huấn luyện model sẽ lâu nhất), nhưng nó lại là bước quyết định sự thành bại của toàn bộ pipeline.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Trước đây tôi nghĩ việc tăng rank LoRA thật cao (ví dụ r=271) sẽ luôn đem lại kết quả tốt hơn rank thấp. Giờ tôi nhận ra rằng vị trí đặt adapter (all-linear phủ toàn bộ text decoder) quan trọng hơn nhiều so với việc tập trung rank lớn vào một vài layer attention duy nhất.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI assistant để viết code xây dựng các hàm kiểm thử và tự động hóa việc chấm điểm qualitative. AI assistant ban đầu đề xuất dùng cờ `assistant_only_loss=True` của TRL, điều này sai hoàn toàn do chat template của Qwen3.5 không hỗ trợ các thẻ đặc biệt để TRL tự sinh mask chính xác.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Bước đầu tiên là xây dựng một bộ dữ liệu test chuẩn, sau đó thiết lập và đo lường một baseline thật mạnh bằng prompt engineering (b) trước khi thực hiện bất kỳ bước huấn luyện nào, nhằm đảm bảo có một cột mốc đối chứng trung thực.
