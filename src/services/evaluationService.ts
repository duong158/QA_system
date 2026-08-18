import type { EvaluationErrorItem, EvaluationMetric, ReaderComparisonRow } from '@/types/qa';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface EvaluationData {
  evaluationMetrics: EvaluationMetric[];
  retrieverChartData: any[];
  recallCurveData: any[];
  readerComparison: ReaderComparisonRow[];
  errorAnalysis: EvaluationErrorItem[];
}

export async function fetchEvaluationData(): Promise<EvaluationData> {
  const response = await fetch(`${API_BASE_URL}/api/evaluation`);
  if (!response.ok) {
    throw new Error('Failed to fetch evaluation data from backend.');
  }
  return response.json();
}
