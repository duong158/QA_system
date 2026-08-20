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

export interface ChatSession {
  id: string;
  title: string;
  turns: ChatTurn[];
  createdAt: number;
  updatedAt: number;
}
