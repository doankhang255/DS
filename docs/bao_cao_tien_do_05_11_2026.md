# Báo cáo tiến độ capstone – 05/11/2026

**Yêu cầu của mentor:** mỗi nhóm báo cáo tiến độ, 5–10 slide, nói tóm tắt trong 5 phút. Học online trên Teams.

Làm **8 slide**, mỗi slide 35–40 giây. Cả bộ đi theo luồng của project: dữ liệu vào từ đâu → xử lý thế nào → nhà đầu tư thấy gì trên web.

## Đang ở đâu so với kế hoạch

Tính từ ngày bắt đầu 15/10/2026, theo lịch 8 tuần trong `docs/FinNews_Tracker_Ke_hoach_trien_khai.pdf`: ngày 02/11 là **cuối tuần 3**, buổi báo cáo 05/11 là **ngày đầu tuần 4**.

| Tuần | Ngày | Mục tiêu theo kế hoạch |
|---|---|---|
| 1 | 15–21/10 | Chốt schema; bắt đầu crawl lại vietstock (có nội dung bài); viết hướng dẫn gán nhãn |
| 2–3 | 22/10–04/11 ← **đang ở đây** | Crawler 2 trang còn lại; cả nhóm gán nhãn theo công ty; đo độ chính xác bộ gắn công ty; dựng DB. Mốc: *dữ liệu vietstock mới đã vào DB* |
| 4–5 | 05–18/11 | Crawl xong 3 trang; ánh xạ nhãn cũ; **train baseline + model theo công ty**; nối web với DB thật |
| 6 | 19–25/11 | Job hằng ngày chạy tự động; thay nhãn baseline bằng model mới |
| 7–8 | 26/11–09/12 | Đánh giá, sửa lỗi, báo cáo, demo |

**Model theo công ty chưa xong là chưa trễ:** theo lịch, việc train model này nằm ở tuần 4–5, hạn khoảng 18/11. Khi báo cáo, trình bày model cấp bài hiện có (E1/E2/E3) làm baseline, kèm kế hoạch cho model theo công ty.

Tình trạng so với mốc cuối tuần 3 (dựa trên repo, ngày 02/11):

| Hạng mục | Tình trạng | Ghi chú |
|---|---|---|
| Gắn công ty (TV3) | **Vượt kế hoạch** | Đã có tên cũ, mã mơ hồ, công ty chính / phụ / nhắc qua và lọc equity news (phân biệt chủ thể vốn là việc của các tuần sau). Còn thiếu: tập ~300 bài để đo precision / recall của bộ gắn công ty (hiện mới có 100 bài đo bước lọc equity), và trích câu ngữ cảnh cho model. |
| Crawler (TV1) | **Đang chạy – vẫn là rủi ro lớn nhất** | `crawl_vietstock_ver2.py` đã lấy được nội dung bài (12 kênh equity). Ngày 03/11: pha danh sách bài được 48.857 bài (01/2010 → 06/2013), pha nội dung chưa chạy. Chưa có crawler diendandoanhnghiep, baodautu. Chưa đạt mốc "dữ liệu vietstock mới vào DB". |
| Database / web (TV2, TV5) | **Chậm** | Mới có kế hoạch web cho 2 người (30/10), chưa có DB hay code `app/`. |
| Gán nhãn theo công ty | Chưa thấy | Trong repo chưa có hướng dẫn gán nhãn hay `entity_labels.csv`. |
| Model theo công ty (TV4) | Đúng lịch | Chưa đến hạn. |

Cần hỏi lại thành viên khác: phần nào đang làm ngoài repo (crawler, DB, gán nhãn) để cập nhật bảng trên.

**Ưu tiên tuần 4:** đẩy crawler (lấy nội dung bài + 2 trang còn lại), vì crawler chặn mọi bước sau; song song bắt đầu gán nhãn theo công ty để kịp train model ở tuần 4–5. Nêu điều này ở slide 8.

## Dàn ý slide

| # | Slide | Nội dung chính | Số liệu / hình |
|---|---|---|---|
| 1 | Đề tài | FinNews Tracker: phân tích cảm xúc tin tài chính Việt Nam theo từng công ty niêm yết. Mở bằng ví dụ "VNM tăng trưởng mạnh, trong khi MSN báo lỗ" → tích cực với VNM, tiêu cực với MSN. | – |
| 2 | Luồng làm việc của cả project | Một sơ đồ từ đầu đến cuối: crawl 3 trang → làm sạch, gộp tin trùng → gắn công ty, lọc equity news → PhoBERT chấm từng cặp (bài, công ty) → database → web app cho nhà đầu tư. Vòng mũi tên "mỗi ngày" quay lại bước crawl. Tô màu: xong / đang làm / chưa làm. | Sơ đồ khối 6 bước + vòng lặp hằng ngày |
| 3 | Dữ liệu: cần gì, đã có gì | **Cần:** tin doanh nghiệp / cổ phiếu của vietstock (12 kênh equity), diendandoanhnghiep, baodautu, **01/2010 → nay**, có nội dung bài; ước tính ~250 nghìn bài riêng vietstock. **Đã crawl:** danh sách 48.857 bài vietstock (01/2010 → 06/2013, đang chạy); chưa tải nội dung; 2 trang còn lại chưa có crawler. Bộ thử 126.576 bài dùng để phát triển code trong lúc chờ. | Bảng 3 nguồn × (phạm vi, ước tính, đã crawl); **cập nhật số mới nhất sáng 05/11** |
| 4 | Gắn công ty & lọc equity news | Luật mã chứng khoán + tên công ty, 395 tên cũ, thương hiệu, tên rút gọn; chọn công ty chủ thể, loại bài chỉ nhắc qua. | 245.363 cặp bài–mã; 85% bài có mã; lọc đúng 96% so với 79% nếu chỉ lọc theo chuyên mục (100 bài gán tay) |
| 5 | Sentiment PhoBERT | E1 (pretrain thêm trên miền tài chính), E2 (chỉ huấn luyện lớp phân loại), E3 (fine-tune) — hiện chấm cấp bài. Bước tiếp: model theo từng công ty, đầu vào "[mã] + các câu nhắc công ty". | **Cần điền:** accuracy / F1 của E1, E2, E3 |
| 6 | Chạy tự động mỗi ngày | Job hằng ngày: lấy bài mới → làm sạch, gộp tin trùng → gắn công ty → chấm sentiment → ghi DB → tính lại xếp hạng mã. Mục tiêu: bài đăng hôm nay lên web ngay trong ngày. | Sơ đồ 6 bước nối nhau + giờ chạy dự kiến |
| 7 | Web app cho nhà đầu tư | Chọn **mã + khoảng ngày + nguồn tin** → số bài tích cực / tiêu cực / trung lập, biểu đồ theo thời gian, danh sách bài có link. Chọn "Tất cả nguồn" thì một tin đăng ở nhiều báo chỉ đếm 1 lần. Trang chủ có **Top mã có tin tức tích cực** (7 / 30 / 90 ngày), kèm dòng "chỉ để tham khảo, không phải khuyến nghị đầu tư". | Mockup 2 màn hình: trang chủ (Top mã) và trang công ty (bộ lọc + biểu đồ) |
| 8 | Tiến độ & việc tiếp theo | Đang ở tuần 4/8. Ưu tiên: crawl xong vietstock và 2 trang còn lại, gán nhãn theo công ty, train model theo công ty trước 18/11, nối web với DB thật. | Lịch 8 tuần, đánh dấu "đang ở đây" |

## Việc cần làm trước buổi báo cáo

- [ ] Lấy kết quả E1 / E2 / E3 của PhoBERT (accuracy, F1, số bài gán nhãn) cho slide 5.
- [ ] Sáng 05/11: lấy số bài đã crawl mới nhất cho slide 3 (crawler đang chạy, số còn tăng).
- [ ] Xác nhận trạng thái từng bước để tô màu trên slide 2 (crawler các báo khác, DB, web app).
- [ ] Vẽ sơ đồ luồng gọn cho slide 2 và 6. Hình chi tiết `Firm_extractor/firm_equity_pipeline.png` quá dày để chiếu, chỉ dùng khi mentor hỏi sâu.
- [ ] Vẽ mockup 2 màn hình web cho slide 7.
- [ ] Hỏi tiến độ thực tế của từng thành viên, cập nhật bảng "Đang ở đâu so với kế hoạch".
- [ ] Chốt mốc thời gian cho slide 7 (crawl chính thức, gán nhãn, web) theo lịch tuần 4–8 ở trên.
- [ ] Phân công ai nói slide nào; tập nói thử, bấm giờ cho đủ 5 phút.
- [ ] Làm slide (chưa làm).

## Chuẩn bị trả lời câu hỏi

- **Vì sao gắn công ty trước rồi mới lọc equity news?** Chỉ lọc theo chuyên mục thì đúng 79%, cách hiện tại đúng 96% trên 100 bài gán tay. Cách làm này giống các nghiên cứu quốc tế: Tetlock et al. (2008), điểm relevance của RavenPack, hệ thống Cassandra của ECB. Chi tiết trong `Firm_extractor/tong_quan_tai_lieu.md`.
- **Đánh giá có đáng tin không?** Mẫu 100 bài còn nhỏ và đã được dùng khi chỉnh luật, nên 96% có thể cao hơn thực tế. Kế hoạch: gán nhãn thêm 100–200 bài ngẫu nhiên.
- **Điểm mới so với nghiên cứu khác?** Chưa thấy nghiên cứu tiếng Việt nào làm đủ hai bước gắn tin với mã niêm yết và lọc equity news. Các bài tiếng Việt đã tìm chỉ làm sentiment.
- **Xếp hạng mã có phải khuyến nghị đầu tư không?** Không. Điểm chỉ dựa trên số bài tích cực / tiêu cực về công ty trong 7 / 30 / 90 ngày: `(tích cực − tiêu cực) / (tổng + 5)`, chỉ tính mã có từ 5 bài. Số 5 giúp mã ít bài không vọt lên đầu. Web luôn ghi rõ "chỉ để tham khảo".
- **Lọc theo nguồn để làm gì?** Một tin hay được nhiều báo đăng lại. Chọn từng nguồn để xem riêng từng báo; chọn "Tất cả" thì tin trùng được gộp, mỗi tin chỉ đếm 1 lần.
- **Lỗi còn lại?** Bỏ sót bài về công ty chưa có trong danh sách mã, ví dụ "Sách Hòa Bình", "HappyMoney". Văn bản viết thường toàn bộ cũng không áp dụng được luật tên còn tiếp tục.
