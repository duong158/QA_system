import { createFallbackResponse, mockKnowledgeBase } from '@/data/mockResponses';
import type { AskQuestionRequest, QaResponse, RetrieverComparisonRow, RetrieverType } from '@/types/qa';

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function normalizeText(value: string): string {
  return value.toLowerCase().normalize('NFD').replace(/\p{Diacritic}/gu, '');
}

function scoreKeywords(question: string, keywords: string[]): number {
  const normalized = normalizeText(question);
  return keywords.reduce((total, keyword) => {
    const keywordNormalized = normalizeText(keyword);
    return normalized.includes(keywordNormalized) ? total + 1 : total;
  }, 0);
}

function pickEntry(question: string) {
  const ranked = mockKnowledgeBase
    .map((entry) => ({ entry, score: scoreKeywords(question, entry.keywords) }))
    .sort((left, right) => right.score - left.score);

  return ranked[0]?.score ? ranked[0].entry : undefined;
}

function trimPassages<T extends { rank: number }>(passages: T[], topK: number): T[] {
  return passages.slice(0, Math.max(1, topK)).map((passage, index) => ({
    ...passage,
    rank: index + 1,
  }));
}

function buildNoAnswerResponse(question: string, topK: number): QaResponse {
  const fallback = createFallbackResponse(question);
  return {
    ...fallback,
    passages: trimPassages(fallback.passages, topK),
  };
}

function remapRetrieverScores(comparison: Record<RetrieverType, RetrieverComparisonRow>, retriever: RetrieverType): RetrieverComparisonRow {
  return comparison[retriever];
}

export async function askQuestionMock(request: AskQuestionRequest): Promise<QaResponse> {
  const entry = pickEntry(request.question);
  await delay(400);
  await delay(500);
  await delay(500);

  if (!entry) {
    return buildNoAnswerResponse(request.question, request.top_k);
  }

  const response = structuredClone(entry.response);
  response.question = request.question;
  response.retriever = request.retriever;
  response.reader = request.reader;
  response.passages = trimPassages(response.passages, request.top_k);
  response.processing_time_ms = 400 + 500 + 500 + Math.round(Math.random() * 60);

  const comparison = remapRetrieverScores(entry.compare, request.retriever);
  response.retriever = comparison.retriever;
  response.processing_time_ms = comparison.responseTimeMs;

  if (!response.answer) {
    response.confidence = 0.32;
  }

  return response;
}

export async function compareRetrieversMock(question: string): Promise<RetrieverComparisonRow[]> {
  const entry = pickEntry(question) ?? mockKnowledgeBase[0];
  await delay(240);
  return Object.values(entry.compare).filter((item) => item.retriever !== 'pyserini' || question.length > 0);
}