# RAG Vietnamese QA (Streamlit)

Dự án RAG (Retrieval-Augmented Generation) cho tài liệu PDF tiếng Việt. Hệ thống gồm các bước: tải dữ liệu PDF, làm sạch văn bản, chia đoạn, xây dựng vector index (Chroma), và trả lời câu hỏi bằng mô hình LLM qua giao diện Streamlit.

## Cấu trúc thư mục

```
Seminar_1/
├─ app/
│  └─ streamlit_app.py
├─ scripts/
│  ├─ benchmark_embeddings.py
│  ├─ build_index.py
│  ├─ debug_rag.py
│  └─ download_data.py
├─ src/
│  ├─ config.py
│  ├─ rag/
│  │  ├─ llm.py
│  │  ├─ loader.py
│  │  ├─ pipeline.py
│  │  ├─ splitter.py
│  │  └─ vector_db.py
│  └─ evaluation/
│     └─ gemini_evaluator.py
├─ data/
├─ storage/
├─ artifacts/
├─ requirements.txt
└─ .env.example
```

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tạo file `.env` từ `.env.example` và điền các biến môi trường cần thiết (HF_TOKEN, GEMINI_API_KEY...).

## Chuẩn bị dữ liệu

Hỗ trợ các định dạng: PDF, TXT, MD, DOCX, XLSX. Với tài liệu dạng bảng, hệ thống trích xuất bảng và chunk theo JSON (row-based) khi bật `INGEST_TABLES=true`. Riêng PDF biểu phí có thể bật `FEE_FACT_CHUNKING=true` để tách theo từng dòng/segment và tạo Q&A theo cấu trúc JSON.

Tải PDF mẫu (từ Google Drive):

```bash
python scripts/download_data.py
```

Hoặc tự đặt tài liệu vào `data/generative_ai/` theo các định dạng trên.

## Xây dựng index

```bash
python scripts/build_index.py
```

Sau khi chạy, vector index sẽ nằm trong `storage/chroma/` và file `artifacts/split_documents.txt` sẽ lưu các chunks (phục vụ debug).

## Chạy ứng dụng Streamlit

```bash
streamlit run app/streamlit_app.py
```

## Đánh giá (tùy chọn)

Nếu muốn chấm điểm chất lượng RAG bằng Gemini, sử dụng class `GeminiRAGEvaluator` trong `src/evaluation/gemini_evaluator.py` và cấu hình `GEMINI_API_KEY`.

Chạy batch evaluation mẫu:

```bash
python scripts/evaluate_batch.py --model gemini-2.5-flash
```

## Lưu ý

- Mô hình LLM và reranker khá nặng; nên chạy trên máy có GPU.
- Nếu không có GPU, có thể đổi sang model nhỏ hơn bằng `LLM_MODEL_NAME` trong `.env`.
- Có thể tắt reranking bằng `USE_RERANKING=false`.
