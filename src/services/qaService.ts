import { askQuestionMock, compareRetrieversMock } from '@/services/mockQaService';
import type { AskQuestionRequest, QaResponse, RetrieverComparisonRow } from '@/types/qa';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const USE_MOCK_API = String(import.meta.env.VITE_USE_MOCK_API ?? 'true').toLowerCase() !== 'false';

function normalizeQaResponse(data: QaResponse): QaResponse {
  return {
    ...data,
    passages: data.passages ?? [],
  };
}

export async function askQuestion(request: AskQuestionRequest): Promise<QaResponse> {
  if (USE_MOCK_API) {
    return askQuestionMock(request);
  }

  const response = await fetch(`${API_BASE_URL}/api/ask`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`API request failed with status ${response.status}`);
  }

  const data = (await response.json()) as QaResponse;
  return normalizeQaResponse(data);
}

export async function compareRetrievers(question: string): Promise<RetrieverComparisonRow[]> {
  if (USE_MOCK_API) {
    return compareRetrieversMock(question);
  }

  const response = await fetch(`${API_BASE_URL}/api/compare`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    throw new Error(`Compare request failed with status ${response.status}`);
  }

  return (await response.json()) as RetrieverComparisonRow[];
}
