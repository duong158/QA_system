export type PipelineState =
  | 'idle'
  | 'listening'
  | 'retrieving'
  | 'reading'
  | 'extracting'
  | 'completed'
  | 'speaking'
  | 'no-answer'
  | 'error';

export type PipelineStepKey = 'question' | 'retriever' | 'reader' | 'answer';
export type PipelineStepStatus = 'waiting' | 'running' | 'completed' | 'error';

export interface PipelineStepModel {
  key: PipelineStepKey;
  label: string;
  status: PipelineStepStatus;
  progress: number;
  accent: 'white' | 'cyan' | 'violet' | 'gold';
}