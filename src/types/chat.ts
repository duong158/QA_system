import type { QaResponse } from './qa';

export type ChatTurnStatus = 'pending' | 'complete' | 'error';

export interface ChatTurn {
  id: string;
  question: string;
  createdAt: number;
  status: ChatTurnStatus;
  response?: QaResponse;
  error?: string;
}
