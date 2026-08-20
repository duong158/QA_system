import type {
  AskQuestionRequest,
  QaResponse,
  RetrieverComparisonRow,
  SocraticFollowUpsRequest,
  SocraticFollowUpsResponse,
} from '@/types/qa';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const QA_ERROR_MESSAGE = 'Không thể xử lý câu hỏi do hệ thống QA gặp lỗi.';
const QA_TIMEOUT_MESSAGE = 'Hệ thống QA xử lý quá lâu. Vui lòng thử lại sau khi Reader đã nạp xong.';
const QA_REQUEST_TIMEOUT_MS = 60_000;

async function readApiError(response: Response): Promise<string> {
  const errorBody = await response.json().catch(() => null) as { error?: string } | null;
  return errorBody?.error || QA_ERROR_MESSAGE;
}

function normalizeQaResponse(data: QaResponse): QaResponse {
  return {
    ...data,
    confidence: data.confidence ?? null,
    answer_confidence: data.answer_confidence ?? null,
    has_answer: data.has_answer ?? Boolean(data.answer),
    selected_passage_id: data.selected_passage_id ?? data.answer_source?.passage_id ?? data.source?.passage_id ?? null,
    source: data.answer_source ?? data.source ?? null,
    passages: data.passages ?? [],
  };
}

export async function askQuestion(request: AskQuestionRequest, signal?: AbortSignal): Promise<QaResponse> {
  let response: Response;
  const controller = new AbortController();
  let timedOut = false;
  const forwardAbort = () => controller.abort();
  signal?.addEventListener('abort', forwardAbort, { once: true });
  const timeout = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, QA_REQUEST_TIMEOUT_MS);
  try {
    response = await fetch(`${API_BASE_URL}/api/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
      signal: controller.signal,
    });
  } catch {
    if (timedOut) {
      throw new Error(QA_TIMEOUT_MESSAGE);
    }
    throw new Error(QA_ERROR_MESSAGE);
  } finally {
    window.clearTimeout(timeout);
    signal?.removeEventListener('abort', forwardAbort);
  }

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }

  const data = (await response.json()) as QaResponse;
  return normalizeQaResponse(data);
}

export async function compareRetrievers(question: string): Promise<RetrieverComparisonRow[]> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/compare`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question }),
    });
  } catch {
    throw new Error(QA_ERROR_MESSAGE);
  }

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }

  return (await response.json()) as RetrieverComparisonRow[];
}

export async function fetchSocraticFollowups(
  request: SocraticFollowUpsRequest,
  signal?: AbortSignal,
): Promise<SocraticFollowUpsResponse> {
  const response = await fetch(`${API_BASE_URL}/api/socratic/followups`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }

  const data = (await response.json()) as Partial<SocraticFollowUpsResponse>;
  return {
    followups: Array.isArray(data.followups) ? data.followups : [],
    processing_time_ms: data.processing_time_ms ?? 0,
    grounding: data.grounding ?? 'selected_and_retrieved_corpus_passages',
    probe: data.probe ?? null,
    debug: data.debug,
  };
}
