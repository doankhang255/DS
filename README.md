# FinNews Tracker

Phân tích cảm xúc tin tức tài chính Việt Nam theo từng công ty niêm yết. Hệ thống crawl bài từ
vietstock.vn, diendandoanhnghiep.vn, baodautu.vn (backfill lịch sử + cập nhật hằng ngày), gắn mỗi bài
với các công ty được nhắc tới, rồi phân loại tích cực / tiêu cực / trung lập **đối với từng công ty**.
Web app cho phép chọn mã + khoảng ngày để xem số bài tích cực, tiêu cực kèm link.

Kế hoạch và phân công: [docs/FinNews_Tracker_Ke_hoach_trien_khai.pdf](docs/FinNews_Tracker_Ke_hoach_trien_khai.pdf).

> **Lưu ý:** bộ 126.576 bài (2010–2023) trong `data_news/` chỉ là **dữ liệu thử** để phát triển code.
> Dữ liệu chính thức sẽ được crawl lại toàn bộ.

## Cấu trúc

```
Crawler/                  crawl_vietstock.py - crawl danh sách + meta bài vietstock.vn
EDA_clean/                làm sạch dữ liệu thô -> corpus tin doanh nghiệp
call_firm_extractor.py    luật tìm mã CK / tên công ty trong 1 bài
run_firm_extractor.py     chạy bộ tìm công ty cho cả corpus
sentiment_label_method/   PhoBERT
  pretrain/               E1 - domain-adaptive pretraining
  model/                  E2 (frozen probe), E3 (fine-tune), common.py
  improve/                repeated CV Tune/Holdout, phương sai theo seed
  inference/              chấm sentiment cả corpus
data_news/                dữ liệu (không đưa lên git, trừ 3 bảng tra cứu nhỏ)
docs/                     tài liệu kế hoạch
```

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Mọi lệnh bên dưới chạy **từ thư mục gốc của repo**; các script tự tìm đường dẫn theo vị trí file,
không phụ thuộc thư mục đang đứng.

## Dữ liệu

Git chỉ chứa `categories.csv`, `domain_norm.csv`, `vn_stock_symbols.csv`. Các file còn lại lấy từ nơi
lưu chung của nhóm rồi đặt đúng chỗ:

| File | Dùng cho |
|---|---|
| `data_news/equity_news_clean_content.parquet` | corpus đã làm sạch (bộ thử) |
| `data_news/GT.csv`, `data_news/Đức_GT.csv` | nhãn tay của 2 người gán |
| `data_news/data_tokenized/equity_news_tokenized_vncorenlp.parquet` | đầu vào PhoBERT |
| `sentiment_label_method/Traditional_ML/Common/tune_holdout/*.csv` | chia Tune/Holdout cho `improve/` |
| `sentiment_label_method/Model_Output/` | checkpoint PhoBERT (base, E1, E3) |

## Chạy

```bash
# Crawl vietstock (mặc định 2010-01-01 -> hôm nay, chạy tiếp được khi bị ngắt)
python Crawler/crawl_vietstock.py --start 2024-01-01 --max-articles 20

# Làm sạch: dataset_news -> clean_news -> equity_news -> equity_news_clean_content
python EDA_clean/news_clean.py
python EDA_clean/to_equity_news.py
python EDA_clean/clean_content.py

# Gắn công ty vào bài -> data_news/firm_extract/
python run_firm_extractor.py --limit 2000      # bỏ --limit để chạy cả corpus

# PhoBERT
python sentiment_label_method/pretrain/domain_adaptive_pretrain.py   # E1
python sentiment_label_method/model/frozen_feature_probe.py          # E2
python sentiment_label_method/model/finetune_phobert.py              # E3
python sentiment_label_method/inference/score_corpus_phobert.py      # chấm cả corpus
```

`EDA_clean/to_equity_news.py` đọc `clean_news_tokenized.parquet`, file này chưa có script nào trong repo
tạo ra; pipeline làm sạch sẽ được gom lại khi crawl dữ liệu chính thức.
