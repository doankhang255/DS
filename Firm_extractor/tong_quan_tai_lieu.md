# Tổng quan tài liệu: lọc tin theo công ty và gắn tên công ty với mã
Các nghiên cứu quốc tế xác nhận hai bước chính của FinNews Tracker: gắn tin với đúng công ty, rồi chỉ giữ tin mà công ty là chủ thể. Tài liệu này tóm tắt những bài báo dùng làm cơ sở cho hai bước đó.

- **Phần 1 – Lọc tin theo công ty:** tiêu chí để coi một bài là tin của một công ty (equity / firm-specific news).
- **Phần 2 – Gắn tên công ty với mã:** cách nhận ra công ty trong văn bản và gắn đúng mã, kể cả khi tên mơ hồ hoặc viết tắt.

Mỗi bài gồm: thông tin xuất bản, nội dung chính, và điểm liên hệ với pipeline (`call_firm_extractor.py`, `filter_equity_news.py`).

## Phần 1 – Lọc tin theo công ty (equity / firm-specific news)

Các nghiên cứu đều cho thấy: phần lớn tin có nhắc tên công ty không phải tin về công ty đó. Chỉ khi lọc ra tin mà công ty là chủ thể, mối liên hệ giữa tin và giá cổ phiếu mới hiện rõ.

### 1.1. Tetlock, Saar-Tsechansky & Macskassy (2008) – More Than Words

[More Than Words: Quantifying Language to Measure Firms' Fundamentals](https://business.columbia.edu/sites/default/files-efs/pubfiles/3096/More_Than_Words_tetlock.pdf), *Journal of Finance* 63(3), 2008.

- **Dữ liệu:** tin của Wall Street Journal và Dow Jones News Service về các công ty trong S&P 500, lấy từ Factiva.
- **Tiêu chí chọn tin của một công ty** (đã đối chiếu với bản PDF):
  - tên chính thức xuất hiện ít nhất 1 lần trong 25 chữ đầu, tính cả tiêu đề;
  - tên thường gọi được nhắc ít nhất 2 lần trong cả bài;
  - bài dài từ 50 chữ trở lên.
- **Kết quả:** tỷ lệ từ tiêu cực trong tin công ty dự báo được lợi nhuận kế toán và lợi suất cổ phiếu.
- **Liên hệ với pipeline:** đây là cơ sở trực tiếp cho luật chọn công ty chính trong `filter_equity_news.py`. Luật của mình là mã ở tiêu đề hoặc sapo (`LEAD_POSITIONS`), hoặc được nhắc từ 2 lần (`SUBJECT_MENTIONS = 2`).

"
* **Lập danh sách công ty và mã định danh tài chính:** Tác giả lấy danh sách toàn bộ các công ty nằm trong chỉ số **S&amp;P 500** giai đoạn 1980–2004 từ cơ sở dữ liệu **CRSP**[1]. Mỗi công ty đều được liên kết trực tiếp với các mã định danh tài chính như **PERMNO** (mã cổ phiếu trong CRSP), **CUSIP**, hoặc **GVKEY** (trong Compustat) để ghép nối dữ liệu giá và báo cáo tài chính[1].
* **Xác định từ khóa tìm kiếm (Tên công ty):** Từ tên công ty trong dữ liệu CRSP (bao gồm cả file theo dõi lịch sử đổi tên công ty)[3], tác giả kết hợp 4 nguồn (trang web S&amp;P, Mergent Online, SEC và Factiva) để xác định chính xác các tên thông dụng (*common/popular names*) mà báo chí thường dùng[2].
* **Truy vấn và trích xuất tự động trên Factiva:** Tác giả xây dựng một **hệ thống trích xuất tự động** (*automated story retrieval system*)[2]. Hệ thống này tự động gửi các câu lệnh truy vấn chứa chuỗi tên công ty lên cơ sở dữ liệu **Factiva** để tải về các bài báo từ *Wall Street Journal* và *Dow Jones News Service*[2].
* **Lọc bài viết theo các tiêu chí đã nêu:** Sau khi tải về, các bài báo mới được chạy qua các bộ lọc tiêu chí (như tên chính thức xuất hiện trong 25 từ đầu, tên phổ biến xuất hiện ít nhất 2 lần, độ dài từ 50 từ trở lên...) nhằm đảm bảo bài báo thực sự tập trung vào công ty đó chứ không phải là bảng biểu hay danh sách tin vặt[6]."

-> ở bài báo đầu tiên tôi muốn bám sát các bước này 

* Chỉ dùng tên công ty mẹ / công ty niêm yết:** Tác giả tập trung vào tên của công ty holding (công ty niêm yết trực tiếp gắn vớ cổ phiếu), **không sử dụng tên của các công ty con hay thương hiệu sản phẩm (ý này cũng hay)

* Tên pháp lý của công ty trong cơ sở dữ liệu chứng khoán CRSP thường bị viết tắt rất khó tra trên báo chí (ví dụ: `PAN AMERN WORLD AWYS INC`). Vì vậy, tác giả kết hợp **4 nguồn dữ liệu** để tìm ra "tên thường gọi" (*common/popular names*) trên báo chí[2]:

1. **Website S&amp;P (S&amp;P Constituents):** Trích xuất tên thông dụng cho 473 công ty thuộc S&amp;P 500 tại thời điểm cuối năm 2004[2].
2. **Mergent Online:** Tra cứu tên thường dùng cho 336 công ty giai đoạn sau năm 1993[3].
3. **Cơ sở dữ liệu SEC:** Tra cứu cho 20 công ty giai đoạn sau 1993 nhưng thiếu thông tin trên Mergent[3].
4. **Tính năng tìm kiếm của Factiva:** Tìm tên báo chí hay dùng cho 285 công ty thuộc giai đoạn trước năm 1993[3]. 


(kiểm tra lại 25 từ đầu tiên còn tuỳ, vì đây là bản tin tức tài chính tiếng việt, bài dài từ 50 chữ trở lên cũng cần có căn cứ, tên thường gọi 2 lần trong bài điều này có thể là hợp lí và kiểm tra xem Factiva có giúp ích được gì cho dự án này không)

### 1.2. Boudoukh, Feldman, Kogan & Richardson (2013) – Which News Moves Stock Prices

[Which News Moves Stock Prices? A Textual Analysis](https://www.nber.org/papers/w18725), NBER Working Paper 18725, 2013 ([bản PDF](https://www.stern.nyu.edu/sites/default/files/assets/documents/con_038156.pdf)).

- **Dữ liệu:** hơn 475.000 tin Dow Jones Newswire về các công ty S&P 500, giai đoạn 2000–2009.
- **Phương pháp:**
  - Phân loại tin theo sự kiện: 14 nhóm, 56 nhóm con.
  - Chủ yếu dùng luật (không phải học máy), chạy thử và chỉnh trên một tập bài huấn luyện.
- **Kết quả:**
  - Chỉ 30% số tin được xác định là sự kiện liên quan đến công ty.
  - R² của mô hình thị trường là 15% vào ngày có tin đã nhận diện, so với 34% vào ngày không có tin. Nghĩa là vào ngày có tin, giá biến động theo thông tin riêng của công ty nhiều hơn.
  - Sau biến động mạnh: ngày không có tin thì giá đảo chiều khoảng 80 điểm cơ bản, ngày có tin đã nhận diện thì giá tiếp tục đi cùng chiều.
- **Liên hệ với pipeline:** ủng hộ việc loại bài "nhắc qua" thay vì giữ mọi bài có mã. Còn cho thấy luật viết tay, chỉnh trên mẫu có nhãn, là cách làm được chấp nhận.

### 1.3. Điểm relevance của RavenPack

Mô tả theo [Babii, Ball, Ghysels & Striaukas (2023), Panel Data Nowcasting](https://arxiv.org/pdf/2307.02673), arXiv 2307.02673.

- **Định nghĩa:** RavenPack là nhà cung cấp dữ liệu tin tức phổ biến nhất trong nghiên cứu tài chính. Mỗi cặp (tin, công ty) có điểm relevance từ 0 đến 100, đo mức độ gắn của tin với công ty: 0 là chỉ nhắc qua, 100 là công ty là chủ thể.
- **Ngưỡng phổ biến:** chỉ giữ tin có relevance từ 75 trở lên, theo hướng dẫn của RavenPack và cách làm của giới thực hành.
- **Liên hệ với pipeline:** các nhãn "công ty chính", "công ty phụ", "nhắc qua" của mình là bản rời rạc của điểm này. Nếu cần một điểm liên tục, có thể tính từ `position` và `mentions` đã có trong `firm_symbols.csv`.

### 1.4. Jeon, McCurdy & Zhao (2022) – News as Sources of Jumps

[News as sources of jumps in stock returns: Evidence from 21 million news articles for 9000 companies](https://econpapers.repec.org/RePEc:eee:jfinec:v:145:y:2022:i:2:p:1-17), *Journal of Financial Economics* 145(2), 2022.

- **Dữ liệu:** khoảng 21 triệu bài báo về hơn 9.000 công ty niêm yết.
- **Kết quả:**
  - Những cú nhảy giá cổ phiếu gắn chặt với số lượng và nội dung tin, và mối liên hệ này mạnh dần qua các thập kỷ.
  - Độ nhạy khác nhau giữa các loại tin.
- **Liên hệ với pipeline:** cho thấy việc gắn tin với công ty và phân loại loại tin (`article_type`) đều có ích cho bước phân tích sau.

## Phần 2 – Gắn tên công ty với mã (entity linking / NER tài chính)

Khó khăn chính là báo ít khi dùng tên pháp lý đầy đủ. Tên viết tắt, tên cũ hay tên thương hiệu lại dễ trùng với công ty khác hoặc với từ thường. Các bài dưới đây mô tả đúng những vấn đề mà `call_firm_extractor.py` đang xử lý.

### 2.1. Repke & Krestel – Extraction and Representation of Financial Entities from Text

[Extraction and Representation of Financial Entities from Text](https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/publications/PDFs/2021_repke_extraction.pdf), chương sách tổng quan, Hasso Plattner Institute, 2021.

- **Nội dung:** tổng quan các bộ dữ liệu, công cụ NER và cách gắn thực thể tài chính (công ty, quan hệ giữa công ty) vào đồ thị tri thức.
- **Ba khó khăn của entity linking:**
  1. **Tên nhiều biến thể:** "VW" và "Volkswagen"; Porsche có nhiều cách viết tên pháp lý.
  2. **Tên mơ hồ:** một cách gọi ứng với nhiều thực thể. "VW" có thể là Volkswagen hoặc Vorwerk; Wikidata có gần 80 mục khi tìm "Volkswagen".
  3. **Thực thể không có trong danh sách:** công ty được nhắc nhưng chưa có trong cơ sở dữ liệu.
- **Tham khảo thêm trong chương:** Loster et al., "Improving company recognition from unstructured text by using dictionaries". Bài này nhận công ty bằng từ điển tên, dùng cấu trúc cây (trie) để xác định đúng điểm đầu, điểm cuối của tên, gần với cách khớp tên dài nhất của mình.
- **Liên hệ với pipeline:**
  - Khó khăn 1 tương ứng cột `old_names`, `brand_name` và luật tên rút gọn.
  - Khó khăn 2 tương ứng tên dùng chung nhiều mã, mã "mơ hồ", và luật mã kèm tên sàn trong ngoặc.
  - Khó khăn 3 tương ứng công ty chưa niêm yết, nằm trong `firm_unmatched.csv`.

### 2.2. Xia, Lin, Liu & Lau (2014) – Linking Financial News Articles to Company Short Names

[Leaning to Train: Linking Financial News Articles to Company Short Names](https://scholars.cityu.edu.hk/en/publications/leaning-to-train-linking-financial-news-articles-to-company-short/), IEEE ICEBE 2014, tr. 240–245.

- **Bài toán:** báo thường gọi công ty bằng tên ngắn, và tên ngắn có thể mơ hồ ("apple" là Apple Inc. hay quả táo). Cần quyết định một lần nhắc có phải tên ngắn của công ty hay không.
- **Phương pháp:**
  - Không dùng học có giám sát, vì gán nhãn tốn kém và tên ngắn mới xuất hiện liên tục.
  - Dùng topic model để tự thu thập dữ liệu huấn luyện.
- **Kết quả:** gần bằng phương pháp có giám sát tốt nhất huấn luyện trên dữ liệu người gán nhãn.
- **Liên hệ với pipeline:** cùng bài toán với luật tên rút gọn và tên thương hiệu một chữ (ví dụ "Vincom" là VIC hay VRE). Hiện mình xử lý bằng luật; cách của bài này là hướng thay thế nếu muốn học từ ngữ cảnh.

### 2.3. Balashov & Nikiforov (2019) – Name/Ticker Confusion

[How much do investors trade because of name/ticker confusion?](https://ideas.repec.org/a/eee/finmar/v46y2019ics1386418118303094.html), *Journal of Financial Markets* 46, 2019.

- **Nội dung:** tìm các cặp công ty có tên hoặc mã giống nhau trên thị trường Mỹ.
- **Kết quả:**
  - 12–25% số cặp có khối lượng giao dịch biến động cùng nhau vì nhà đầu tư nhầm.
  - Khoảng 5% khối lượng giao dịch là do nhầm.
  - Khi tên hoặc mã đổi, hiện tượng này biến mất.
- **Liên hệ với pipeline:** bằng chứng rằng trùng tên, trùng mã là vấn đề thật, cả với con người. Đây là lý do cần luật cho mã "mơ hồ" và tên dùng chung.

### 2.4. Kouris et al. (2025) – Cassandra (ECB)

[Predicting bank's distress using AI and NLP on financial news](https://www.suomenpankki.fi/globalassets/bof/en/financial-stability/events/cassandra-an-automated-aibased-surveillance-system-to-forecast-bank-distress-from-financial-news.pdf), Ngân hàng Trung ương châu Âu (ECB), trình bày tại hội thảo của Ngân hàng Trung ương Phần Lan, 2025.

- **Dữ liệu:** hơn 15 triệu tin từ 2007 đến 2023, gắn với 191 ngân hàng.
- **Thứ tự các bước:**
  1. NER gắn mỗi tin với một hoặc nhiều ngân hàng, tách tin của từng ngân hàng khỏi tin theo quốc gia, khu vực, ngân hàng trung ương.
  2. Bỏ tin không liên quan bằng luật n-gram, ví dụ tin "BofA hạ khuyến nghị cổ phiếu X" có nhắc tên ngân hàng nhưng không nói về ngân hàng đó.
  3. Bỏ tin trùng bằng thuật toán đo độ giống nhau.
  4. Chấm sentiment, rồi dùng làm đặc trưng dự báo. Mô hình dự báo ngân hàng gặp khó khăn đạt độ chính xác 85%.
- **Liên hệ với pipeline:** đây là hệ thống thực tế có thứ tự giống FinNews Tracker nhất: gắn công ty → lọc tin liên quan bằng luật → sentiment. Luật loại tin "môi giới khuyến nghị" của họ cũng tương tự luật 1.3b "Chứng khoán X" của mình.

## Đối chiếu luật của pipeline với tài liệu

Mỗi luật chính của pipeline đều có ít nhất một bài để dẫn.

| Luật trong pipeline | File | Bài để dẫn |
| --- | --- | --- |
| Công ty chính: mã ở tiêu đề/sapo, hoặc nhắc ≥ 2 lần | `filter_equity_news.py` | Tetlock et al. (2008) |
| Phân biệt công ty chính / phụ / nhắc qua | `filter_equity_news.py` | Điểm relevance RavenPack (ngưỡng 75) |
| Loại bài chỉ nhắc qua công ty | `filter_equity_news.py` | Boudoukh et al. (2013): chỉ 30% tin là sự kiện liên quan |
| Phân loại loại tin (`article_type`) | `filter_equity_news.py` | Boudoukh et al. (2013); Jeon et al. (2022) |
| Luật viết tay, chỉnh trên mẫu có nhãn (`label_equity_100.csv`) | cả hai | Boudoukh et al. (2013) |
| Tên cũ, thương hiệu, tên rút gọn | `call_firm_extractor.py` | Repke & Krestel; Xia et al. (2014) |
| Khớp tên dài nhất theo từ điển | `call_firm_extractor.py` | Loster et al. (trong Repke & Krestel) |
| Mã "mơ hồ", tên dùng chung nhiều mã | `call_firm_extractor.py` | Repke & Krestel; Balashov & Nikiforov (2019) |
| Gắn công ty trước, lọc tin sau, rồi sentiment | cả hai | Kouris et al. (2025) |

Mình chưa tìm thấy nghiên cứu nào cho tiếng Việt làm đủ hai bước này. Các bài tiếng Việt đã tìm chỉ làm sentiment bằng PhoBERT, nên đây có thể là điểm đóng góp của đề tài.

## Tài liệu tham khảo

1. Tetlock, P. C., Saar-Tsechansky, M., & Macskassy, S. (2008). More Than Words: Quantifying Language to Measure Firms' Fundamentals. *Journal of Finance*, 63(3). [PDF](https://business.columbia.edu/sites/default/files-efs/pubfiles/3096/More_Than_Words_tetlock.pdf)
2. Boudoukh, J., Feldman, R., Kogan, S., & Richardson, M. (2013). Which News Moves Stock Prices? A Textual Analysis. NBER Working Paper 18725. [NBER](https://www.nber.org/papers/w18725)
3. Babii, A., Ball, R. T., Ghysels, E., & Striaukas, J. (2023). Panel Data Nowcasting: The Case of Price-Earnings Ratios. arXiv 2307.02673. [arXiv](https://arxiv.org/pdf/2307.02673)
4. Jeon, Y., McCurdy, T., & Zhao, X. (2022). News as sources of jumps in stock returns: Evidence from 21 million news articles for 9000 companies. *Journal of Financial Economics*, 145(2), 1–17. [EconPapers](https://econpapers.repec.org/RePEc:eee:jfinec:v:145:y:2022:i:2:p:1-17)
5. Repke, T., & Krestel, R. (2021). Extraction and Representation of Financial Entities from Text. Hasso Plattner Institute. [PDF](https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/publications/PDFs/2021_repke_extraction.pdf)
6. Xia, Y., Lin, H., Liu, Y., & Lau, R. (2014). Leaning to Train: Linking Financial News Articles to Company Short Names. *IEEE ICEBE 2014*, 240–245. [CityU Scholars](https://scholars.cityu.edu.hk/en/publications/leaning-to-train-linking-financial-news-articles-to-company-short/)
7. Balashov, V. S., & Nikiforov, A. (2019). How much do investors trade because of name/ticker confusion? *Journal of Financial Markets*, 46. [IDEAS](https://ideas.repec.org/a/eee/finmar/v46y2019ics1386418118303094.html)
8. Kouris, A., Arranz Barquero, J., Parafioriti, L., & Petropoulos, A. (2025). Predicting bank's distress using AI and NLP on financial news (Cassandra). ECB. [PDF](https://www.suomenpankki.fi/globalassets/bof/en/financial-stability/events/cassandra-an-automated-aibased-surveillance-system-to-forecast-bank-distress-from-financial-news.pdf)
