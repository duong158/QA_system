# VIQA Nexus

VIQA Nexus là hệ thống hỏi đáp tiếng Việt chạy cục bộ, kết hợp truy hồi tài liệu, Reader trích xuất XLM-RoBERTa và mô hình sinh LFM2.5. Giao diện được xây dựng bằng React, TypeScript và Vite; backend là HTTP API viết bằng Python.

Project hiện hỗ trợ hai hướng trả lời:

- **Extractive QA:** tìm passage phù hợp rồi trích xuất span bằng checkpoint `xlm-roberta-large-viquad`.
- **Generative QA:** dùng LFM2.5-2.6B ở chế độ RAG hoặc hội thoại trực tiếp.

Ngoài hỏi đáp, hệ thống còn có gợi ý câu hỏi tiếp theo, so sánh retriever, phản hồi có kiểm duyệt, quản lý tài liệu đóng góp và trang phân tích điểm mù tri thức.

## Trạng thái hiện tại

| Thành phần | Trạng thái |
| --- | --- |
| Corpus runtime | 5.317 tài liệu, 6.544 passage đã chia theo câu trong `docs.db` |
| Retriever | TF-IDF, BM25, Dense và Hybrid |
| Extractive Reader | XLM-RoBERTa tại `models/reader/xlm-roberta-large-viquad` |
| Local LLM | `LFM/LFM2.5-2.6B`, nạp bằng Transformers và bitsandbytes 4-bit |
| Gợi ý Socratic | Luôn được gọi sau câu trả lời; truy hồi câu hỏi gần nhất bằng BM25 |
| Feedback | SQLite, có quy trình chờ duyệt; không tự huấn luyện model lúc runtime |
| Frontend | React 18, TypeScript, Vite, Zustand, Tailwind CSS, Three.js |

> Tên lựa chọn `phobert` vẫn được giữ trong API và local storage để tương thích với UI cũ. Ở cấu hình hiện tại, lựa chọn này cũng trỏ tới checkpoint XLM-RoBERTa trong `config/qa_pipeline.json`, không phải trọng số PhoBERT cũ.

## Kiến trúc tổng quát

```text
Người dùng
    |
    v
React/Vite frontend (localhost:5173)
    |
    v
Python HTTP API (localhost:8000)
    |
    +--> TF-IDF / BM25 / Dense / Hybrid
    |          |
    |          v
    |    data/processed/docs.db
    |
    +--> XLM-RoBERTa Extractive Reader
    |
    +--> LFM2.5 Local LLM (RAG hoặc Direct)
    |
    +--> Socratic BM25 + Feedback SQLite + Evaluation API
```

Luồng extractive thực hiện truy hồi, chạy Reader trên các passage ứng viên, chấm điểm lại bằng retrieval/Reader/answer type và áp dụng các cổng từ chối trước khi trả lời. Luồng LFM có hai chế độ:

| Reader gửi từ frontend | Cách hoạt động |
| --- | --- |
| `phobert` | Khóa tương thích cũ; hiện dùng checkpoint XLM-RoBERTa đã cấu hình |
| `xlmr` | Gọi trực tiếp XLM-RoBERTa extractive QA |
| `llm` | Dùng tối đa 5 passage làm RAG khi retrieval đủ mạnh; nếu không thì trả lời trực tiếp |
| `llm_chat` | LFM trả lời trực tiếp, không gắn nguồn tài liệu |

Frontend mặc định dùng **BM25**, Reader `xlmr` và `top_k = 10`. Nếu gọi `/api/ask` mà không truyền retriever, backend dùng **Hybrid** theo `config/qa_pipeline.json`.

## Cấu trúc thư mục chính

```text
QA_system/
├── backend/                    # API, LFM Reader, Socratic và feedback
├── config/                     # Cấu hình pipeline và semantic policy
├── data/
│   ├── processed/docs.db       # Corpus SQLite dùng khi chạy
│   ├── raw/                    # Các split ViQuAD dạng Parquet
│   └── feedback/               # CSDL feedback sinh lúc runtime
├── models/reader/
│   └── xlm-roberta-large-viquad/
├── reader/                     # Huấn luyện, dự đoán và đánh giá Reader
├── retrieval/                  # Các retriever và script tạo index
├── src/                        # Frontend React/TypeScript
├── tests/                      # Kiểm thử backend
├── report.tex                  # Báo cáo kỹ thuật
├── SLIDE_CONTENT.md            # Dàn nội dung thuyết trình
└── package.json
```

## Yêu cầu môi trường

- Python 3.10 trở lên; project hiện được kiểm tra với Python 3.12.0.
- Node.js 18 trở lên; môi trường hiện tại dùng Node.js 22.16.0 và npm 10.9.2.
- Dung lượng trống cho checkpoint XLM-R khoảng 2,2 GB và cache LFM.
- GPU NVIDIA/CUDA được khuyến nghị mạnh khi chạy LFM2.5. Code hiện nạp LFM bằng lượng tử hóa 4-bit của bitsandbytes; máy chỉ có CPU hoặc driver không tương thích có thể không nạp được LLM.
- Kết nối mạng ở lần đầu tải LFM hoặc Dense Encoder từ Hugging Face. Sau khi model đã nằm trong cache có thể chạy offline.

## Cài đặt

Các lệnh dưới đây dành cho Windows PowerShell và được chạy tại thư mục gốc của project:

```powershell
cd D:\GitHub\QA_system

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements-reader.txt
python -m pip install -r requirements-retrieval.txt

npm install
```

Để dùng Dense Retrieval đầy đủ, cài thêm:

```powershell
python -m pip install sentence-transformers
```

Nếu Dense Encoder không khả dụng, Hybrid runtime sẽ giảm cấp về BM25 thay vì làm hỏng toàn bộ yêu cầu.

### Dữ liệu bắt buộc

Backend cần tệp sau để khởi động:

```text
data/processed/docs.db
```

Gợi ý câu hỏi tiếp theo đọc các tệp:

```text
data/raw/viquad2_train.parquet
data/raw/viquad2_validation.parquet
data/raw/viquad2_test.parquet
```

Ở trạng thái hiện tại, ba split cung cấp khoảng 39.446 câu hỏi duy nhất cho chỉ mục Hybrid gợi ý.

### Checkpoint XLM-RoBERTa

Thư mục sau phải tồn tại:

```text
models/reader/xlm-roberta-large-viquad/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

`model.safetensors` có dung lượng lớn và bị loại khỏi Git bởi `.gitignore`, vì vậy clone repository mới sẽ không tự có trọng số này. Cần sao chép checkpoint vào đúng đường dẫn trước khi chạy Extractive Reader.

### LFM

Mặc định backend dùng model:

```text
LFM2.5-2.6B
```

Lần chạy đầu tiên cần tải model vào Hugging Face cache. Khi model đã được cache đầy đủ, có thể buộc Transformers chỉ dùng dữ liệu cục bộ:

```powershell
$env:HF_HUB_OFFLINE="1"
```

Không bật biến này ở lần đầu tải model.

## Hướng dẫn chạy

### Cách 1: Chạy đầy đủ XLM-RoBERTa và LFM2.5-2.6B

Mở **hai cửa sổ PowerShell**.

Terminal 1 — backend:

```powershell
cd D:\GitHub\QA_system
.\.venv\Scripts\Activate.ps1

$env:QA_PRELOAD_READER="true"
$env:QA_PRELOAD_LLM="true"
python backend\viqa_api.py
```

Backend chỉ bắt đầu nhận request sau khi cả hai model đã nạp xong. Lần đầu nạp LFM có thể mất nhiều thời gian vì phải tải model.

Terminal 2 — frontend:

```powershell
cd D:\GitHub\QA_system
$env:VITE_API_BASE_URL="http://localhost:8000"
npm run dev
```

Mở đúng địa chỉ:

```text
http://localhost:5173/
```

Không thêm `/HTTP` vào cuối URL. Các trang frontend hiện có:

- `http://localhost:5173/` — giao diện hỏi đáp.
- `http://localhost:5173/evaluation` — kết quả đánh giá đã lưu.
- `http://localhost:5173/knowledge-blind-spots` — phản hồi, duyệt dữ liệu và phân tích điểm mù.

Kiểm tra backend:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Trường `loaded_readers` nên có `llm` và ít nhất một trong `phobert`/`xlmr` sau khi preload hoàn tất.

### Cách 2: Chỉ chạy XLM-RoBERTa

Cấu hình này nhẹ hơn và không nạp LFM khi khởi động:

```powershell
$env:QA_PRELOAD_READER="true"
$env:QA_PRELOAD_LLM="false"
python backend\viqa_api.py
```

Nếu sau đó chọn LFM trên giao diện, model vẫn được nạp lười ở request đầu tiên.

### Cách 3: Chỉ preload LFM2.5

```powershell
$env:QA_PRELOAD_READER="false"
$env:QA_PRELOAD_LLM="true"
python backend\viqa_api.py
```

XLM-RoBERTa vẫn có thể được nạp lười nếu người dùng chuyển sang Reader extractive.

### Chạy với thiết lập mặc định

Lệnh npm sau tương đương chạy trực tiếp backend với `QA_PRELOAD_READER=true` và `QA_PRELOAD_LLM=false` nếu chưa đặt biến môi trường:

```powershell
npm run api
```

> Hiện không khuyến nghị dùng `run_system.bat`: tệp này còn gắn cứng đường dẫn `D:\Python\python.exe` và chưa thiết lập LFM. Quy trình hai terminal ở trên phản ánh đúng runtime hiện tại.

## Sử dụng giao diện

- Nhấn **Enter** để gửi câu hỏi.
- Nhấn **Shift + Enter** để xuống dòng.
- Có thể nhập giọng nói tiếng Việt nếu trình duyệt hỗ trợ Web Speech API.
- Mở phần thiết lập để chọn Retriever, Reader, `top_k`, giọng đọc và hiệu ứng hiển thị.
- Lịch sử giữ tối đa 50 phiên trong local storage của trình duyệt.
- Thời gian chờ tối đa của một request hỏi đáp ở frontend là 180 giây để model cục bộ có đủ thời gian xử lý.

## Gợi ý câu hỏi Socratic

Sau khi có câu trả lời, frontend tự gọi `POST /api/socratic/followups`; giao diện không còn nút bật/tắt chế độ Gia sư.

Implementation đang chạy nằm ở `backend/related_questions.py`:

1. Đọc câu hỏi từ ba split ViQuAD train/validation/test.
2. Tạo chỉ mục BM25 ở lần gọi đầu tiên.
3. Lấy tối đa 3 câu hỏi gần với câu hiện tại.
4. Loại câu trùng và các câu đã hỏi trong phiên.

## API chính

| Phương thức | Endpoint | Chức năng |
| --- | --- | --- |
| `GET` | `/health` | Trạng thái corpus, model, retriever và cấu hình |
| `POST` | `/api/ask` | Hỏi đáp bằng Reader được chọn |
| `POST` | `/api/compare` | So sánh TF-IDF, BM25, Dense và Hybrid |
| `POST` | `/api/socratic/followups` | Lấy tối đa 3 câu hỏi liên quan |
| `GET` | `/api/evaluation` | Đọc artifact đánh giá trong thư mục `results` |
| `POST` | `/api/feedback` | Gửi phản hồi cho một câu trả lời |
| `GET` | `/api/feedback/analytics` | Thống kê feedback và điểm mù |
| `GET` | `/api/feedback/review` | Danh sách feedback chờ duyệt |
| `POST` | `/api/feedback/{id}/review` | Duyệt hoặc từ chối feedback |
| `GET/POST` | `/api/documents/submissions` | Xem hoặc gửi tài liệu đóng góp |
| `POST` | `/api/documents/submissions/{id}/review` | Duyệt tài liệu đóng góp |

Ví dụ gọi API bằng PowerShell:

```powershell
$body = @{
  question = "Phạm Văn Đồng là ai?"
  retriever = "bm25"
  reader = "xlmr"
  top_k = 10
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://localhost:8000/api/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## Feedback và vòng lặp tri thức

Feedback được lưu ở:

```text
data/feedback/feedback.db
```

Các bản ghi đi qua trạng thái `PENDING`, `REVIEWED`, `APPROVED` hoặc `REJECTED`. Hệ thống xác minh correction span theo passage corpus trước khi lưu. Tuy nhiên:

- Feedback không được dùng trực tiếp để trả lời câu hỏi.
- Model không tự fine-tune trong lúc chạy.
- Tài liệu được duyệt chỉ trở thành ứng viên; corpus production không tự cập nhật.
- Endpoint review chưa có authentication hoặc phân quyền, chỉ phù hợp môi trường local/demo.

Luồng cải tiến dự kiến:

```text
Feedback đã duyệt
    -> xuất dataset
    -> huấn luyện offline
    -> đánh giá benchmark
    -> con người phê duyệt
    -> triển khai checkpoint mới
```

Xuất dữ liệu và báo cáo feedback:

```powershell
python export_feedback_dataset.py
python evaluate_feedback_loop.py
```

## Biến môi trường

### Backend

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `QA_HOST` | `0.0.0.0` | Địa chỉ backend lắng nghe |
| `QA_PORT` | `8000` | Cổng API |
| `QA_PRELOAD_READER` | `true` | Nạp XLM-RoBERTa trước khi mở server |
| `QA_PRELOAD_LLM` | `false` | Nạp LFM trước khi mở server |
| `QA_LLM_MODEL` | `LFM/LFM2.5-1.7B` | Model ID hoặc đường dẫn local của LLM |
| `QA_LLM_MAX_NEW_TOKENS` | `96` | Số token sinh tối đa, tối thiểu 16 |
| `QA_DENSE_MODEL` | `keepitreal/vietnamese-sbert` | Dense Encoder |
| `QA_DEBUG` | `false` | In log kỹ thuật của pipeline |
| `HF_HUB_OFFLINE` | không đặt | Chỉ dùng cache Hugging Face khi bằng `1` |

Các trọng số reranking, ngưỡng Reader, kích thước chunk và checkpoint production nằm trong `config/qa_pipeline.json`.

### Frontend

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `VITE_API_BASE_URL` | `http://localhost:8000` | Địa chỉ backend |
| `VITE_QA_DEBUG` | `false` | Hiện thêm điểm kỹ thuật ngoài development |
| `VITE_AVATAR_MODEL_URL` | `/models/mari.vrm` | Đường dẫn avatar VRM |
| `VITE_AVATAR_MODEL_NAME` | `Mari 3D VRoid Model` | Tên avatar |
| `VITE_AVATAR_CREATOR_NAME` | `wondrous21` | Tên tác giả avatar |
| `VITE_AVATAR_LICENSE` | `Free to use with credit` | Thông tin giấy phép hiển thị |

`VITE_USE_MOCK_API` còn xuất hiện trong `.env.example` để tương thích cấu hình cũ nhưng frontend hiện không đọc biến này và luôn gọi API thật.

## Build và kiểm thử

Build frontend production:

```powershell
npm run build
```

Xem thử bản build:

```powershell
npm run preview
```

Chạy toàn bộ test Python:

```powershell
python -m pytest -q
```

Kiểm tra integrity của span ViQuAD:

```powershell
python -m reader.validate_spans --splits train validation
```

Đánh giá retriever và QA:

```powershell
python evaluate_retriever.py path\to\eval.jsonl --method bm25 --k 1 3 5 10
python evaluate_qa.py --validation --mode oracle
python evaluate_qa.py --validation --mode end-to-end --retriever bm25 --top-k 10
```

Tại lần kiểm tra gần nhất ngày 21/08/2026, `npm run build` thành công; bộ test Python có **207 test đạt và 15 test lỗi**. Vì vậy repository hiện chưa ở trạng thái test xanh hoàn toàn và không nên coi các artifact đánh giá cũ là benchmark của LFM/XLM-R hiện tại nếu chưa chạy lại đúng checkpoint.

## Xử lý lỗi thường gặp

### Cổng 8000 đang được sử dụng

Thông báo:

```text
Cannot start VIQA API on 0.0.0.0:8000: the port is already in use.
```

Tìm process đang giữ cổng:

```powershell
Get-NetTCPConnection -LocalPort 8000 |
  Select-Object -ExpandProperty OwningProcess -Unique
```

Kiểm tra PID trước khi dừng:

```powershell
Get-Process -Id <PID>
Stop-Process -Id <PID>
```

Hoặc đổi cổng cho cả backend và frontend:

```powershell
# Terminal backend
$env:QA_PORT="8001"
python backend\viqa_api.py

# Terminal frontend
$env:VITE_API_BASE_URL="http://localhost:8001"
npm run dev
```

### Frontend báo không thể nhận câu trả lời

1. Mở `http://localhost:8000/health` hoặc chạy `Invoke-RestMethod` để kiểm tra API.
2. Xem terminal backend có đang nạp model, thiếu checkpoint hay hết bộ nhớ không.
3. Đảm bảo `VITE_API_BASE_URL` đúng cổng và khởi động lại Vite sau khi đổi biến.
4. Request đầu tiên có thể chậm do model được nạp lười; frontend chờ tối đa 180 giây.

### Không có gợi ý câu hỏi

- Cài `rank-bm25`, pandas và pyarrow bằng `requirements-retrieval.txt`.
- Kiểm tra ba tệp Parquet trong `data/raw`.
- Gợi ý chỉ được gọi khi câu trả lời chính có nội dung.
- Xem log `[QuestionBM25Index]` ở terminal backend khi endpoint được gọi lần đầu.

### LFM không nạp được

- Kiểm tra CUDA, driver NVIDIA, PyTorch và bitsandbytes có tương thích không.
- Đảm bảo máy còn đủ VRAM/RAM và dung lượng cache.
- Không đặt `HF_HUB_OFFLINE=1` khi model chưa được tải đủ.
- Có thể tạm chạy extractive-only bằng `QA_PRELOAD_LLM=false`.

### Thiếu checkpoint XLM-R

Đảm bảo `model.safetensors` nằm đúng tại:

```text
models/reader/xlm-roberta-large-viquad/model.safetensors
```

Nếu không có checkpoint này, Reader extractive không thể nạp và backend sẽ trả lỗi rõ ràng; runtime hiện không âm thầm coi sentence fallback là model production thay thế.

## Tài liệu dự án

- Báo cáo kỹ thuật: `report.tex` và `output/pdf/VIQA_Nexus_Report.pdf`.
- Nội dung slide: `SLIDE_CONTENT.md`.
- Cấu hình pipeline: `config/qa_pipeline.json`.
- Cấu hình semantic policy: `config/semantic_policy.json`.

Avatar Mari không được commit cùng repository. Nếu có quyền sử dụng model VRM, đặt tệp tại `public/models/mari.vrm` hoặc đổi `VITE_AVATAR_MODEL_URL` sang đường dẫn phù hợp.
