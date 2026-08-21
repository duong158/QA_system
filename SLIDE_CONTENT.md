# NỘI DUNG SLIDE BÁO CÁO — VIQA NEXUS

Tài liệu này là storyboard để dựng slide bảo vệ project. Phần **Nội dung trên slide**
có thể đưa trực tiếp vào PowerPoint; phần **Gợi ý trực quan** dùng để chọn hình, biểu đồ
hoặc bố cục; dòng **Nguồn** nên đặt nhỏ ở chân slide hoặc trong speaker notes.

Đối tượng trình bày: giảng viên và sinh viên có kiến thức nền về NLP.

Thông điệp xuyên suốt: **VIQA Nexus đã phát triển từ một pipeline
Retriever–Reader thành hệ thống QA tiếng Việt lai XLM-R–Qwen3 có truy vết nguồn,
giao diện hội thoại và vòng phản hồi; tuy nhiên model active, Socratic grounding và
độ ổn định kiểm thử vẫn cần hoàn thiện trước khi triển khai production.**

---

## Slide 1 — VIQA Nexus: Hệ thống hỏi đáp tiếng Việt lai XLM-R–Qwen3

### Nội dung trên slide

- **VIQA Nexus**
- Hệ thống hỏi đáp tiếng Việt trên kho tài liệu đóng
- Kiến trúc lai: Extractive QA · RAG · Direct Chat
- Nhóm 15 — NLP Summer 2026

### Gợi ý trực quan

Trang tiêu đề tối giản, dùng logo UET và một ảnh chụp giao diện VIQA Nexus làm nền mờ.

Nguồn: report.tex, src/pages/HomePage.tsx

---

## Slide 2 — Người dùng cần câu trả lời, không chỉ một danh sách tài liệu

### Nội dung trên slide

- Tìm kiếm truyền thống trả về nhiều tài liệu để người dùng tự đọc.
- QA cần trả lời trực tiếp và chỉ ra bằng chứng đã sử dụng.
- Với tiếng Việt, chất lượng truy hồi, span trả lời và xử lý câu không có đáp án đều
  là thách thức.
- Hệ thống cần cân bằng giữa **khả năng trả lời** và **khả năng từ chối khi thiếu bằng
  chứng**.

### Gợi ý trực quan

So sánh hai luồng đơn giản:

**Câu hỏi → Danh sách tài liệu** và
**Câu hỏi → Câu trả lời + Nguồn**.

Nguồn: Chương 1–2 của report.tex

---

## Slide 3 — Mục tiêu là một hệ thống QA có thể kiểm tra được quyết định

### Nội dung trên slide

- Hỗ trợ nhiều Retriever và nhiều chế độ Reader.
- Bảo toàn liên kết từ câu trả lời đến document, passage và bằng chứng.
- Dùng semantic gates để giảm câu trả lời sai quan hệ hoặc sai tiền đề.
- Tạo trải nghiệm hội thoại với gợi ý câu hỏi tiếp theo và phản hồi người dùng.
- Không tự động học từ dữ liệu chưa được kiểm duyệt.

### Gợi ý trực quan

Dùng năm cụm từ lớn quanh tên hệ thống:
**Retrieval · Answering · Evidence · Tutoring · Feedback**.

Nguồn: config/qa_pipeline.json, backend/viqa_api.py, backend/feedback.py

---

## Slide 4 — Một Retriever chung phục vụ hai đường trả lời khác nhau

### Nội dung trên slide

~~~text
Người dùng
    ↓
React UI → HTTP API → Retriever
                         ├─ XLM-R → Span candidates → Semantic gates
                         └─ Qwen3 → RAG hoặc Direct generation
                              ↓
                    Câu trả lời + nguồn nếu có
~~~

- Lớp Socratic và Feedback chạy sau đường trả lời chính.
- Lỗi gợi ý hoặc analytics không làm hỏng /api/ask.

### Gợi ý trực quan

Dùng duy nhất một sơ đồ kiến trúc ngang. Nhấn màu khác nhau cho nhánh extractive và
nhánh generative.

Nguồn: backend/viqa_api.py, Hình 4.1 trong report.tex

---

## Slide 5 — Corpus runtime gồm 5.317 tài liệu và 6.544 passage

### Nội dung trên slide

| Thành phần | Quy mô |
|---|---:|
| Train questions | 28.454 |
| Validation questions | 3.814 |
| Test questions | 7.301 |
| Documents trong docs.db | 5.317 |
| Passage sau sentence-aware chunking | 6.544 |

- Passage tối đa 220 token, chồng lấn 2 câu.
- Chunking theo biên câu giúp giảm rủi ro đáp án bị cắt ở ranh giới đoạn.

### Gợi ý trực quan

Biểu đồ dòng dữ liệu:
**ViQuAD2 parquet → clean corpus → docs.db → sentence-aware passages**.

Nguồn: data/raw/, data/processed/docs.db, backend/chunking.py

---

## Slide 6 — Bốn Retriever cho bốn điểm cân bằng khác nhau

### Nội dung trên slide

- **TF–IDF:** nhẹ, dễ giải thích, phụ thuộc từ khóa.
- **BM25:** lexical ranking tốt, là mặc định của UI.
- **Dense:** tìm tương đồng ngữ nghĩa bằng Vietnamese SBERT.
- **Hybrid:** hợp nhất BM25 và Dense bằng Reciprocal Rank Fusion.

> Backend mặc định Hybrid; UI khởi tạo BM25 để giảm cold start trên máy demo.

### Gợi ý trực quan

Dùng một trục từ “lexical” đến “semantic”, đặt bốn phương pháp lên trục; không dùng
bốn card giao diện.

Nguồn: backend/viqa_api.py, config/qa_pipeline.json, src/store/appStore.ts

---

## Slide 7 — Hybrid đạt Recall@10 cao nhất trong benchmark Retriever

### Nội dung trên slide

| Retriever | MRR | Recall@1 | Recall@10 |
|---|---:|---:|---:|
| BM25 | 0,6856 | 0,5946 | 0,8507 |
| TF–IDF | 0,6512 | 0,5582 | 0,8142 |
| Dense | 0,7321 | 0,6654 | 0,8923 |
| **Hybrid** | **0,7682** | **0,6981** | **0,9156** |

- Hybrid tăng 6,49 điểm phần trăm Recall@10 so với BM25.
- Đây là benchmark document-level trên 7.301 câu hỏi test, không phải chất lượng QA
  end-to-end.

### Gợi ý trực quan

Dùng biểu đồ đường Recall@k từ IMG/report_retriever_recall.png; đặt một callout
“Recall@10 = 91,56%”.

Nguồn: results/retriever_eval_all.json

---

## Slide 8 — XLM-R trích xuất câu trả lời và cho phép hệ thống từ chối

### Nội dung trên slide

- Checkpoint active: models/reader/xlm-roberta-large-viquad.
- Reader sinh tối đa 5 span neural cho mỗi passage.
- Phrase fallback và sentence fallback chỉ là phương án có penalty.
- Mỗi candidate phải vượt các gate về:
  - span và boundary;
  - completeness và evidence;
  - subject, relation và answer type;
  - final ranking.

### Gợi ý trực quan

Trình bày chuỗi gate như một đường kiểm tra liên tiếp, không dùng sơ đồ mạng phức tạp.

Nguồn: reader/predict.py, reader/candidate_validation.py,
reader/relation_validator.py

---

## Slide 9 — Semantic gating giảm trả lời sai nhưng làm tăng nguy cơ từ chối đúng

### Nội dung trên slide

| Cấu hình | Overall F1 | Unanswerable Accuracy | FP rate | FN rate |
|---|---:|---:|---:|---:|
| Reader only | 24,63 | 22,50% | 77,50% | 15,00% |
| Full pipeline | **26,56** | **57,50%** | **42,50%** | 44,09% |

- Full pipeline giảm false positive rõ rệt.
- Đổi lại, false negative tăng: hệ thống an toàn hơn nhưng bảo thủ hơn.
- Checksum semantic holdout hiện lệch file lock, nên kết quả chỉ là diagnostic tạm thời.

### Gợi ý trực quan

Dùng IMG/report_semantic_ablation.png và tô nổi hai cột Reader only/Full pipeline.

Nguồn: results/semantic_holdout_v1_report_refactored_v2_final.json

---

## Slide 10 — Qwen3 bổ sung khả năng tổng hợp và hội thoại cục bộ

### Nội dung trên slide

- Model mặc định: **Qwen3 1.7B**, lượng tử NF4 4-bit.
- **reader=llm**:
  - RAG trên tối đa 5 passage khi retrieval score đủ mạnh;
  - chuyển sang direct answer khi tín hiệu truy hồi dưới ngưỡng 0,30.
- **reader=llm_chat:** luôn trả lời trực tiếp, không gắn nguồn.
- XLM-R và Qwen3 có thể cùng tồn tại trong bộ nhớ.

> Giá trị confidence 1,0 của Qwen hiện là placeholder, không phải xác suất đúng.

### Gợi ý trực quan

Chia slide thành hai nhánh chữ lớn: **RAG — có context/source** và
**Direct Chat — linh hoạt nhưng không source**.

Nguồn: backend/llm_reader.py, backend/viqa_api.py

---

## Slide 11 — Smoke test xác nhận cả XLM-R và Qwen3 đều chạy thật

### Nội dung trên slide

| Chế độ | Câu hỏi kiểm tra | Thời gian |
|---|---|---:|
| Qwen direct | Thủ đô của Việt Nam là gì? | 7,55 s |
| Qwen RAG | Phạm Văn Đồng là ai? | 48,52 s |
| XLM-R | Phạm Văn Đồng là ai? | 13,91 s |
| Alias phobert | Cùng checkpoint XLM-R | 15,63 s |

- Health sau kiểm tra ghi nhận llm, phobert, xlmr cùng được nạp.
- Đây là một lần chạy trên host hiện tại, không phải SLA hoặc benchmark chất lượng.

### Gợi ý trực quan

Dùng biểu đồ thanh thời gian; đặt chú thích rõ “single-run smoke test”.

Nguồn: runtime test ngày 21/08/2026, Bảng 10.3 trong report.tex

---

## Slide 12 — Nguồn được chuẩn hóa ở backend, không chỉ bị che bằng CSS

### Nội dung trên slide

**Trước**

> Nguồn: Phạm Văn Đồng (1 tháng 3 năm 1906...

**Sau**

> Nguồn: Phạm Văn Đồng  
> Evidence: Phạm Văn Đồng (1 tháng 3 năm 1906 – 29 tháng 4 năm 2000) là...

- Ưu tiên metadata title → heading → entity đầu passage → fallback câu đầu.
- Chỉ loại phần ngoặc khi nó giống ngày sinh–mất hoặc khoảng năm.

### Gợi ý trực quan

Dùng một hình before/after với chính ví dụ Phạm Văn Đồng.

Nguồn: backend/source_titles.py, tests/test_source_titles.py

---

## Slide 13 — “Gia sư” hiện là bộ gợi ý câu hỏi liên quan luôn bật

### Nội dung trên slide

- Sau mỗi câu trả lời, frontend tự động gọi /api/socratic/followups.
- Backend lập BM25 trên **39.446 câu hỏi ViQuAD duy nhất**.
- Loại câu hiện tại và các câu đã hỏi, sau đó trả tối đa 3 câu gần nhất.
- Nút bật/tắt đã được loại bỏ; lỗi gợi ý không làm lỗi câu trả lời chính.

### Gợi ý trực quan

Hiển thị một chuỗi hội thoại có một câu trả lời và ba quick-reply bên dưới.

Nguồn: src/hooks/useSocraticFollowups.ts, backend/related_questions.py

---

## Slide 14 — Nguyên nhân gốc: gợi ý đang tìm câu giống, không kiểm tra đáp án

### Nội dung trên slide

- BM25 hiện xếp hạng **question text**, không xếp hạng evidence passage.
- Candidate không có source_passage_id, answerability score hoặc QA verification.
- Vì vậy câu hỏi có thể liên quan về từ ngữ nhưng không có đáp án trong tài liệu hiện
  tại.
- Lịch sử chủ thể cũng có thể reset khi toàn bộ câu hỏi thay đổi.

**Hướng sửa đúng:** source retrieval → QA verification → chỉ giữ câu có passage và
span hợp lệ.

### Gợi ý trực quan

Một sơ đồ nhân quả ngắn:

**Question similarity ≠ Evidence grounding → Gợi ý có thể không trả lời được**.

Nguồn: backend/related_questions.py, backend/socratic.py,
src/hooks/useSocraticFollowups.ts

---

## Slide 15 — Giao diện mới biến pipeline thành một cuộc hội thoại

### Nội dung trên slide

- React 18 · TypeScript · Vite · Tailwind · Zustand.
- Bố cục thích ứng: lịch sử — hội thoại — Mari 3D.
- Enter để gửi, Shift+Enter xuống dòng, hỗ trợ IME tiếng Việt.
- Speech Recognition/Synthesis và avatar được tách khỏi logic QA.
- Lưu tối đa 50 session; timeout request QA là 180 giây.

### Gợi ý trực quan

Dùng ảnh chụp toàn màn hình UI hiện tại và đánh số ba vùng chính. Không chèn quá nhiều
ảnh giao diện nhỏ.

Nguồn: src/pages/HomePage.tsx, src/components/chat/ChatComposer.tsx,
src/services/qaService.ts, src/store/appStore.ts

---

## Slide 16 — Feedback được thu thập nhưng không tự học trong runtime

### Nội dung trên slide

~~~text
User feedback
    ↓
SQLite review queue
    ↓
Human review
    ↓
Approved export
    ↓
Offline train + benchmark + promotion
~~~

- Hỗ trợ correct, incorrect, span correction và hai loại lỗi no-answer.
- Deduplicate, conflict detection và review lifecycle.
- runtime_model_updated=false; production corpus không đổi khi submit feedback.
- Hiện mới có 1 feedback thật đang chờ duyệt — chưa đủ để kết luận điểm mù.

### Gợi ý trực quan

Dùng một pipeline tuyến tính 5 bước; nhấn mạnh “Human review” ở trung tâm.

Nguồn: backend/feedback.py, backend/feedback_analytics.py,
data/feedback/feedback.db

---

## Slide 17 — API thống nhất extractive QA, RAG, tutoring và feedback

### Nội dung trên slide

| Endpoint | Vai trò |
|---|---|
| GET /health | Model hỗ trợ/đã nạp, passage, config |
| POST /api/ask | XLM-R extractive hoặc Qwen RAG/direct |
| POST /api/compare | So sánh Retriever |
| POST /api/socratic/followups | Câu hỏi liên quan bằng BM25 |
| POST /api/feedback | Ghi nhận phản hồi |
| GET /api/feedback/analytics | Knowledge blind spots |

- Backend hiện dùng ThreadingHTTPServer, phù hợp demo cục bộ.
- Chưa có authentication, OpenAPI, worker queue hoặc rate limiting.

### Gợi ý trực quan

Chỉ dùng bảng endpoint; không cần ảnh code.

Nguồn: backend/viqa_api.py

---

## Slide 18 — Build thành công nhưng test suite chưa xanh

### Nội dung trên slide

| Kiểm tra ngày 21/08/2026 | Kết quả |
|---|---|
| Frontend build | Thành công — 3.353 module |
| Bundle | Main 794,92 kB; Mari 1.033,46 kB |
| Pytest | **207 passed, 15 failed** |
| Thời gian test | 150,63 giây |

Nhóm failure hiện tại:

- contract frontend/Socratic cũ;
- semantic false premise, cause, location, fallback;
- candidate reranking và config;
- checksum semantic holdout.

### Gợi ý trực quan

Đặt “207 passed” và “15 failed” thành hai số lớn; bên dưới là bốn nhóm lỗi.

Nguồn: npm run build, python -m pytest -q, ngày 21/08/2026

---

## Slide 19 — Project đã chạy được end-to-end nhưng chưa production-ready

### Nội dung trên slide

**Điểm mạnh**

- Bốn Retriever, hai kiểu trả lời và source trace cho nhánh grounded.
- XLM-R và Qwen3 chạy cục bộ, không phụ thuộc API LLM bên ngoài.
- Semantic trace, feedback review và UI hội thoại đã tích hợp.

**Giới hạn quyết định**

- Chưa có full-validation artifact cho XLM-R và factuality benchmark cho Qwen3.
- Qwen direct/chat không có source; confidence chưa hiệu chuẩn.
- Socratic chưa source-verified.
- Test suite, holdout integrity, security và hiệu năng chưa đạt production.

### Gợi ý trực quan

Dùng bố cục cân bằng hai cột “Đã có” và “Còn thiếu”, không dùng màu xanh/đỏ tuyệt đối
để tránh tạo cảm giác pass/fail đơn giản.

Nguồn: Chương 10–11 của report.tex

---

## Slide 20 — Ưu tiên tiếp theo là grounding và khả năng tái lập

### Nội dung trên slide

1. **P0 — Test và holdout:** sửa regression, khôi phục checksum/provenance.
2. **P0 — Socratic grounding:** bắt buộc source passage và QA verification.
3. **P0 — Evaluation:** sinh artifact đúng cho XLM-R và Qwen3.
4. **P1 — Security:** authentication, RBAC, audit và rate limit.
5. **P2 — Performance:** model worker, cache, batching và code splitting.

**Kết luận:** VIQA Nexus là nền tảng QA lai có thể kiểm toán; bước tiếp theo không chỉ
là tăng F1 mà là làm cho mọi câu trả lời và gợi ý đều có bằng chứng, metric và trạng
thái kiểm chứng rõ ràng.

### Gợi ý trực quan

Dùng roadmap từ P0 đến P2 và kết thúc bằng một câu kết luận lớn. Không cần thêm slide
“Cảm ơn” riêng; có thể đặt “Q&A” nhỏ ở chân slide.

Nguồn: Bảng 11.1 trong report.tex

---

# SLIDE DỰ PHÒNG

## Slide A1 — Cấu hình pipeline active

### Nội dung trên slide

| Tham số | Giá trị |
|---|---:|
| API default Retriever | Hybrid |
| UI default Retriever | BM25 |
| top_k mặc định / tối đa | 10 / 20 |
| Candidate pool | max(20, 2k), tối đa 40 |
| Ranking weights | 0,30 / 0,60 / 0,10 / 0,00 |
| Reader window / stride | 256 / 80 |
| Chunk size / overlap | 220 token / 2 câu |
| Extractive checkpoint | XLM-R large ViQuAD |

Nguồn: config/qa_pipeline.json

---

## Slide A2 — Cách chạy đồng thời XLM-R và Qwen3

### Nội dung trên slide

~~~powershell
pip install -r requirements-reader.txt
$env:QA_PRELOAD_READER="true"
$env:QA_PRELOAD_LLM="true"
python backend\viqa_api.py
~~~

Terminal frontend:

~~~powershell
npm install
npm run dev
~~~

- Frontend: http://localhost:5173/
- Health: http://localhost:8000/health
- run_system.bat hiện hard-code Python path không tồn tại và chưa bật Qwen.

Nguồn: requirements-reader.txt, backend/viqa_api.py, run_system.bat

---

## Slide A3 — Demo flow đề xuất

### Nội dung trên slide

1. Mở /health để chỉ ra loaded_readers.
2. Hỏi “Phạm Văn Đồng là ai?” bằng XLM-R và mở source passage.
3. Đổi sang Qwen RAG, hỏi lại để so cách diễn đạt và latency.
4. Chọn một câu gợi ý để minh họa Socratic luôn bật.
5. Giải thích vì sao gợi ý hiện chưa được source-verify.
6. Gửi feedback và mở trang Knowledge Blind Spots.

### Lưu ý khi demo

- Khởi động/preload model trước buổi trình bày.
- Không chạy một backend thứ hai trên cổng 8000.
- Chuẩn bị ảnh/video dự phòng nếu máy demo không đủ RAM hoặc Qwen cold start lâu.

Nguồn: backend/viqa_api.py, src/pages/HomePage.tsx

---

## Slide A4 — Tài liệu tham khảo chính

### Nội dung trên slide

- D. Chen et al. — *Reading Wikipedia to Answer Open-Domain Questions*, ACL 2017.
- S. Robertson và H. Zaragoza — *BM25 and Beyond*, 2009.
- G. Cormack et al. — *Reciprocal Rank Fusion*, SIGIR 2009.
- A. Conneau et al. — *Unsupervised Cross-lingual Representation Learning at
  Scale*, ACL 2020.
- A. Yang et al. — *Qwen3 Technical Report*, arXiv:2505.09388, 2025.
- K. V. Nguyen et al. — *UIT-ViQuAD 2.0*, LREC 2022.

Nguồn: phần Tài liệu tham khảo trong report.tex

