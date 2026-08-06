import type { EvaluationErrorItem, EvaluationMetric, ReaderComparisonRow, RetrieverType } from '@/types/qa';

export const evaluationMetrics: EvaluationMetric[] = [
  { label: 'Recall@1', value: 0.842, comparison: { tfidf: 0.702, bm25: 0.791, dense: 0.842, pyserini: 0.809 } },
  { label: 'Recall@3', value: 0.914, comparison: { tfidf: 0.831, bm25: 0.877, dense: 0.914, pyserini: 0.893 } },
  { label: 'Recall@5', value: 0.947, comparison: { tfidf: 0.884, bm25: 0.914, dense: 0.947, pyserini: 0.921 } },
  { label: 'Recall@10', value: 0.976, comparison: { tfidf: 0.931, bm25: 0.948, dense: 0.976, pyserini: 0.958 } },
  { label: 'MRR', value: 0.881, comparison: { tfidf: 0.731, bm25: 0.826, dense: 0.881, pyserini: 0.845 } },
  { label: 'Exact Match', value: 0.764 },
  { label: 'F1 Score', value: 0.851 },
  { label: 'Avg. Response', value: 0.482 },
];

export const retrieverChartData = [
  { name: 'TF-IDF', recall1: 0.702, recall3: 0.831, recall5: 0.884, recall10: 0.931 },
  { name: 'BM25', recall1: 0.791, recall3: 0.877, recall5: 0.914, recall10: 0.948 },
  { name: 'Dense', recall1: 0.842, recall3: 0.914, recall5: 0.947, recall10: 0.976 },
  { name: 'Pyserini', recall1: 0.809, recall3: 0.893, recall5: 0.921, recall10: 0.958 },
];

export const recallCurveData = [
  { k: '1', tfidf: 0.702, bm25: 0.791, dense: 0.842, pyserini: 0.809 },
  { k: '3', tfidf: 0.831, bm25: 0.877, dense: 0.914, pyserini: 0.893 },
  { k: '5', tfidf: 0.884, bm25: 0.914, dense: 0.947, pyserini: 0.921 },
  { k: '10', tfidf: 0.931, bm25: 0.948, dense: 0.976, pyserini: 0.958 },
];

export const readerComparison: ReaderComparisonRow[] = [
  { reader: 'mock', exactMatch: 0.71, f1: 0.79, avgLatencyMs: 110 },
  { reader: 'phobert', exactMatch: 0.84, f1: 0.89, avgLatencyMs: 248 },
  { reader: 'vibert', exactMatch: 0.81, f1: 0.87, avgLatencyMs: 235 },
  { reader: 'xlmr', exactMatch: 0.83, f1: 0.88, avgLatencyMs: 267 },
];

export const errorAnalysis: EvaluationErrorItem[] = [
  { issue: 'Retriever không đúng passage', count: 18, note: 'Phần lớn xảy ra ở câu hỏi mơ hồ hoặc thuật ngữ đồng nghĩa.' },
  { issue: 'Reader chọn sai answer span', count: 11, note: 'Thường xuất hiện khi câu chứa nhiều mốc số liệu.' },
  { issue: 'Câu trả lời quá dài', count: 7, note: 'Mô hình trả lại cả ngữ cảnh thay vì cụm ngắn.' },
  { issue: 'Câu trả lời quá ngắn', count: 5, note: 'Thiếu cụm định danh chính xác.' },
  { issue: 'Confidence thấp', count: 9, note: 'Nên ưu tiên hiển thị cảnh báo và nguồn.' },
  { issue: 'Không có đáp án', count: 12, note: 'Cần thông báo rõ và gợi ý kiểm tra nguồn khác.' },
];

export const retrieverLabels: Record<RetrieverType, string> = {
  tfidf: 'TF-IDF',
  bm25: 'BM25',
  dense: 'Dense Retrieval',
  pyserini: 'Pyserini BM25',
};