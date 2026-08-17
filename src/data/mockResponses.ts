import type { QaResponse, ReaderType, RetrieverComparisonRow, RetrieverType } from '@/types/qa';

export interface MockKnowledgeEntry {
  keywords: string[];
  response: QaResponse;
  compare: Record<RetrieverType, RetrieverComparisonRow>;
}

const sharedPassages = [
  {
    rank: 1,
    document_id: 'DOC001',
    passage_id: 'DOC001_P03',
    title: 'Quy chế đào tạo',
    page: 12,
    text: 'Sinh viên phải tham gia ít nhất 80% số tiết của học phần để đủ điều kiện dự thi.',
    retrieval_score: 0.923,
    reader_score: 0.914,
  },
  {
    rank: 2,
    document_id: 'DOC001',
    passage_id: 'DOC001_P07',
    title: 'Quy chế đào tạo',
    page: 15,
    text: 'Sinh viên vắng học phải thực hiện thủ tục xin phép theo quy định của cơ sở đào tạo.',
    retrieval_score: 0.761,
    reader_score: 0.402,
  },
  {
    rank: 3,
    document_id: 'DOC002',
    passage_id: 'DOC002_P02',
    title: 'Hướng dẫn học vụ',
    page: 6,
    text: 'Điều kiện dự thi được xác định dựa trên mức độ tham gia học tập và kết quả đánh giá quá trình.',
    retrieval_score: 0.694,
    reader_score: 0.288,
  },
];

const examPassages = [
  {
    rank: 1,
    document_id: 'DOC010',
    passage_id: 'DOC010_P01',
    title: 'Quy định thi cuối kỳ',
    page: 4,
    text: 'Sinh viên được tham dự thi nếu hoàn thành đầy đủ các yêu cầu học phần và không vi phạm kỷ luật.',
    retrieval_score: 0.914,
    reader_score: 0.882,
  },
  {
    rank: 2,
    document_id: 'DOC010',
    passage_id: 'DOC010_P05',
    title: 'Quy định thi cuối kỳ',
    page: 8,
    text: 'Kết quả học tập được tổng hợp từ chuyên cần, bài tập, giữa kỳ và thi kết thúc học phần.',
    retrieval_score: 0.833,
    reader_score: 0.341,
  },
  {
    rank: 3,
    document_id: 'DOC021',
    passage_id: 'DOC021_P02',
    title: 'Sổ tay sinh viên',
    page: 19,
    text: 'Người học cần kiểm tra lịch thi và phòng thi trước ít nhất 24 giờ.',
    retrieval_score: 0.662,
    reader_score: 0.251,
  },
];

const libraryPassages = [
  {
    rank: 1,
    document_id: 'DOC030',
    passage_id: 'DOC030_P02',
    title: 'Thư viện số',
    page: 2,
    text: 'Sinh viên có thể mượn tài liệu trong vòng 14 ngày và gia hạn tối đa 2 lần nếu không có người đặt trước.',
    retrieval_score: 0.899,
    reader_score: 0.863,
  },
  {
    rank: 2,
    document_id: 'DOC031',
    passage_id: 'DOC031_P01',
    title: 'Quy trình mượn trả',
    page: 11,
    text: 'Tài liệu quá hạn sẽ bị ghi nhận và tính phí theo số ngày chậm trả.',
    retrieval_score: 0.723,
    reader_score: 0.314,
  },
  {
    rank: 3,
    document_id: 'DOC030',
    passage_id: 'DOC030_P08',
    title: 'Thư viện số',
    page: 7,
    text: 'Người dùng được truy cập kho tài nguyên học liệu bằng tài khoản sinh viên do trường cấp.',
    retrieval_score: 0.621,
    reader_score: 0.299,
  },
];

const noAnswerPassages = [
  {
    rank: 1,
    document_id: 'DOC900',
    passage_id: 'DOC900_P01',
    title: 'Tập tài liệu chung',
    page: 1,
    text: 'Tài liệu này mô tả các nguyên tắc học tập và hướng dẫn tra cứu nội bộ.',
    retrieval_score: 0.581,
    reader_score: 0.182,
  },
  {
    rank: 2,
    document_id: 'DOC901',
    passage_id: 'DOC901_P03',
    title: 'Ghi chú nội bộ',
    page: 3,
    text: 'Một số câu hỏi cần được kiểm tra thêm từ nguồn chính thức của nhà trường.',
    retrieval_score: 0.553,
    reader_score: 0.144,
  },
  {
    rank: 3,
    document_id: 'DOC902',
    passage_id: 'DOC902_P02',
    title: 'Văn bản tham chiếu',
    page: 5,
    text: 'Không có thông tin cụ thể về chủ đề này trong tập chỉ mục hiện tại.',
    retrieval_score: 0.512,
    reader_score: 0.102,
  },
];

function buildComparison(
  correctPassageRank: number,
  denseScore: number,
  bm25Score: number,
  tfidfScore: number,
  denseTime: number,
  bm25Time: number,
  tfidfTime: number,
  preview: string,
): Record<RetrieverType, RetrieverComparisonRow> {
  return {
    tfidf: {
      retriever: 'tfidf',
      label: 'TF-IDF',
      correctPassageRank,
      recallAt1: correctPassageRank === 1,
      recallAt3: correctPassageRank <= 3,
      responseTimeMs: tfidfTime,
      topPassagePreview: preview,
      retrievalScore: tfidfScore,
    },
    bm25: {
      retriever: 'bm25',
      label: 'BM25',
      correctPassageRank,
      recallAt1: correctPassageRank === 1,
      recallAt3: correctPassageRank <= 3,
      responseTimeMs: bm25Time,
      topPassagePreview: preview,
      retrievalScore: bm25Score,
    },
    dense: {
      retriever: 'dense',
      label: 'Dense Retrieval',
      correctPassageRank,
      recallAt1: correctPassageRank === 1,
      recallAt3: correctPassageRank <= 3,
      responseTimeMs: denseTime,
      topPassagePreview: preview,
      retrievalScore: denseScore,
    },
    pyserini: {
      retriever: 'pyserini',
      label: 'Pyserini BM25',
      correctPassageRank,
      recallAt1: correctPassageRank === 1,
      recallAt3: correctPassageRank <= 3,
      responseTimeMs: bm25Time + 24,
      topPassagePreview: preview,
      retrievalScore: bm25Score - 0.03,
    },
    hybrid: {
      retriever: 'hybrid',
      label: 'Hybrid (BM25 + Dense)',
      correctPassageRank,
      recallAt1: correctPassageRank === 1,
      recallAt3: correctPassageRank <= 3,
      responseTimeMs: bm25Time + denseTime + 8,
      topPassagePreview: preview,
      retrievalScore: Math.min(1, Math.max(bm25Score, denseScore) + 0.015),
    },
  };
}

export const mockKnowledgeBase: MockKnowledgeEntry[] = [
  {
    keywords: ['nghỉ', 'tiết', 'số tiết', 'đi học', 'chuyên cần'],
    response: {
      question: 'Sinh viên được nghỉ tối đa bao nhiêu phần trăm số tiết?',
      answer: 'Sinh viên được phép nghỉ tối đa 20% số tiết.',
      confidence: 0.914,
      processing_time_ms: 482,
      retriever: 'dense',
      reader: 'phobert',
      source: {
        document_id: 'DOC001',
        passage_id: 'DOC001_P03',
        title: 'Quy chế đào tạo',
        page: 12,
      },
      answer_span: {
        text: '80% số tiết',
        start: 25,
        end: 36,
      },
      passages: sharedPassages,
    },
    compare: buildComparison(1, 0.92, 0.81, 0.72, 480, 512, 402, sharedPassages[0].text),
  },
  {
    keywords: ['thi', 'dự thi', 'học phần', 'kỷ luật'],
    response: {
      question: 'Điều kiện dự thi cuối kỳ là gì?',
      answer: 'Sinh viên được dự thi khi hoàn thành yêu cầu học phần và không vi phạm kỷ luật.',
      confidence: 0.872,
      processing_time_ms: 515,
      retriever: 'bm25',
      reader: 'vibert',
      source: {
        document_id: 'DOC010',
        passage_id: 'DOC010_P01',
        title: 'Quy định thi cuối kỳ',
        page: 4,
      },
      answer_span: {
        text: 'hoàn thành đầy đủ các yêu cầu học phần',
        start: 24,
        end: 62,
      },
      passages: examPassages,
    },
    compare: buildComparison(2, 0.88, 0.84, 0.69, 499, 436, 378, examPassages[0].text),
  },
  {
    keywords: ['mượn', 'thư viện', 'tài liệu', 'gia hạn'],
    response: {
      question: 'Sinh viên được mượn tài liệu trong bao lâu?',
      answer: 'Sinh viên có thể mượn tài liệu trong vòng 14 ngày và gia hạn tối đa 2 lần.',
      confidence: 0.941,
      processing_time_ms: 468,
      retriever: 'pyserini',
      reader: 'xlmr',
      source: {
        document_id: 'DOC030',
        passage_id: 'DOC030_P02',
        title: 'Thư viện số',
        page: 2,
      },
      answer_span: {
        text: '14 ngày',
        start: 34,
        end: 41,
      },
      passages: libraryPassages,
    },
    compare: buildComparison(1, 0.94, 0.88, 0.76, 468, 528, 389, libraryPassages[0].text),
  },
  {
    keywords: ['không', 'khong', 'không có', 'chưa tìm thấy'],
    response: {
      question: 'Câu hỏi này không có câu trả lời rõ ràng trong tài liệu?',
      answer: '',
      confidence: 0.24,
      processing_time_ms: 503,
      retriever: 'dense',
      reader: 'mock',
      source: {
        document_id: 'DOC900',
        passage_id: 'DOC900_P01',
        title: 'Tập tài liệu chung',
        page: 1,
      },
      passages: noAnswerPassages,
    },
    compare: buildComparison(4, 0.37, 0.29, 0.22, 503, 521, 406, noAnswerPassages[0].text),
  },
];

export function createFallbackResponse(question: string): QaResponse {
  return {
    question,
    answer: 'Hệ thống chưa tìm thấy câu trả lời đáng tin cậy trong tập tài liệu.',
    confidence: 0.31,
    processing_time_ms: 540,
    retriever: 'dense',
    reader: 'mock',
    source: {
      document_id: 'DOC900',
      passage_id: 'DOC900_P01',
      title: 'Tập tài liệu chung',
      page: 1,
    },
    passages: noAnswerPassages,
  };
}