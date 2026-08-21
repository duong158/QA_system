import { useState } from 'react';
import { Bot, Check, ChevronDown, Clipboard, Database, SlidersHorizontal, Volume2 } from 'lucide-react';
import { AnswerFeedback } from '@/components/feedback/AnswerFeedback';
import { FollowUpInsights } from '@/components/socratic/FollowUpInsights';
import type { SocraticLoadState } from '@/hooks/useSocraticFollowups';
import type { FollowUpCandidate, PassageResult, QaResponse } from '@/types/qa';

interface AssistantMessageProps {
  response: QaResponse;
  followUps?: FollowUpCandidate[];
  followUpsState?: SocraticLoadState;
  followUpsLatencyMs?: number | null;
  onFollowUpSelect?: (followUp: FollowUpCandidate) => void;
  onFollowUpSpeak?: (followUp: FollowUpCandidate) => void;
  onSpeakAnswer: (text: string) => void;
  onSourceClick: (response: QaResponse) => void;
  showDebug?: boolean;
}

function selectedPassage(response: QaResponse): PassageResult | null {
  return response.passages?.find((passage) => passage.passage_id === response.selected_passage_id)
    ?? response.top_retrieved_passage
    ?? response.passages?.[0]
    ?? null;
}

function SourceButton({ response, onSourceClick }: { response: QaResponse; onSourceClick: (response: QaResponse) => void }) {
  const selected = selectedPassage(response);
  if (!selected && !response.source && !response.passages?.length) return null;
  const sourceTitle = response.answer_source?.title || response.source?.title || selected?.title || 'Tài liệu nguồn';
  const page = response.answer_source?.page ?? response.source?.page ?? selected?.page;

  return (
    <button
      type="button"
      onClick={() => onSourceClick(response)}
      className="group mt-3 flex w-full items-center gap-2 rounded-xl border border-[var(--border)] bg-[var(--surface-subtle)] px-3 py-2.5 text-left transition hover:bg-[var(--surface-muted)]"
    >
      <Database className="h-4 w-4 shrink-0 text-viqa-cyan" />
      <span className="min-w-0 flex-1">
        <span className="block text-[10px] font-semibold uppercase tracking-wide text-[var(--text-muted)]">Nguồn</span>
        <span className="block truncate text-sm font-medium text-[var(--text-primary)]">{sourceTitle}{page ? ` · Trang ${page}` : ''}</span>
      </span>
      <span className="text-xs text-[var(--text-muted)] transition group-hover:text-viqa-cyan">Xem chi tiết</span>
    </button>
  );
}

function PipelineDisclosure({ response, showDebug }: { response: QaResponse; showDebug: boolean }) {
  return (
    <details className="group mt-2">
      <summary className="flex cursor-pointer list-none items-center gap-1.5 py-1 text-xs text-[var(--text-muted)] marker:hidden hover:text-[var(--text-secondary)]">
        <SlidersHorizontal className="h-3.5 w-3.5" />
        Xem cách hệ thống tìm câu trả lời
        <ChevronDown className="h-3.5 w-3.5 transition-transform group-open:rotate-180" />
      </summary>
      <div className="mt-2 grid gap-2 rounded-xl border border-[var(--border)] bg-[var(--surface-subtle)] p-3 text-xs text-[var(--text-secondary)] sm:grid-cols-2">
        <p><span className="font-medium text-[var(--text-primary)]">Tìm kiếm:</span> {response.retriever.toUpperCase()}</p>
        <p><span className="font-medium text-[var(--text-primary)]">Đọc hiểu:</span> {response.reader.toUpperCase()}</p>
        <p><span className="font-medium text-[var(--text-primary)]">Loại câu hỏi:</span> {response.question_type?.join(', ') || '--'}</p>
        <p><span className="font-medium text-[var(--text-primary)]">Quan hệ:</span> {response.semantic_relation || response.question_relation || '--'}</p>
        <p><span className="font-medium text-[var(--text-primary)]">Thời gian:</span> {response.processing_time_ms.toLocaleString('vi-VN')} ms</p>
        <p><span className="font-medium text-[var(--text-primary)]">Trạng thái:</span> {response.has_answer ?? Boolean(response.answer) ? 'Có bằng chứng' : 'Chưa đủ bằng chứng'}</p>
        {showDebug ? <p className="sm:col-span-2">Ranking signal: {response.scores?.ranking?.toFixed(4) ?? '--'} · Reader: {response.scores?.reader?.toFixed(4) ?? '--'}</p> : null}
      </div>
    </details>
  );
}

export function ThinkingMessage() {
  return (
    <div className="flex max-w-[88%] items-start gap-2.5" role="status" aria-live="polite">
      <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-50 text-indigo-600"><Bot className="h-4 w-4" /></span>
      <div className="rounded-2xl rounded-tl-md viqa-glass px-4 py-3 text-sm text-[var(--text-secondary)]">
        <span>Mari đang suy nghĩ</span>
        <span className="ml-2 inline-flex gap-1" aria-hidden="true">
          <span className="typing-dot h-1.5 w-1.5 rounded-full bg-indigo-400" />
          <span className="typing-dot h-1.5 w-1.5 rounded-full bg-indigo-400" />
          <span className="typing-dot h-1.5 w-1.5 rounded-full bg-indigo-400" />
        </span>
      </div>
    </div>
  );
}

export function AssistantMessage({
  response,
  followUps = [],
  followUpsState = 'idle',
  followUpsLatencyMs,
  onFollowUpSelect,
  onFollowUpSpeak,
  onSpeakAnswer,
  onSourceClick,
  showDebug = false,
}: AssistantMessageProps) {
  const [copied, setCopied] = useState(false);
  const hasAnswer = response.has_answer ?? Boolean(response.answer);
  const answerText = response.answer || response.no_answer_reason || 'Mari chưa tìm thấy câu trả lời đủ chắc chắn trong tài liệu hiện có.';

  const copyAnswer = async () => {
    await navigator.clipboard.writeText(answerText);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  };

  return (
    <article className="flex max-w-[94%] items-start gap-2.5 sm:max-w-[88%]">
      <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-50 text-indigo-600" aria-label="Mari">
        <Bot className="h-4 w-4" />
      </span>
      <div className="min-w-0 flex-1">
        <div className="rounded-2xl rounded-tl-md viqa-glass px-4 py-3 shadow-sm sm:px-5 sm:py-4">
          <p className={`whitespace-pre-wrap text-[15px] leading-7 ${hasAnswer ? 'text-[var(--text-primary)]' : 'text-[var(--text-secondary)]'}`}>{answerText}</p>
          <SourceButton response={response} onSourceClick={onSourceClick} />

          <div className="mt-3 flex flex-wrap items-center gap-1 border-t border-[var(--border)] pt-2">
            <button type="button" onClick={() => onSpeakAnswer(answerText)} className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs text-[var(--text-secondary)] hover:bg-[var(--surface-muted)]" aria-label="Đọc câu trả lời">
              <Volume2 className="h-3.5 w-3.5" /> Đọc
            </button>
            <button type="button" onClick={() => void copyAnswer()} className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs text-[var(--text-secondary)] hover:bg-[var(--surface-muted)]" aria-label="Sao chép câu trả lời">
              {copied ? <Check className="h-3.5 w-3.5 text-emerald-600" /> : <Clipboard className="h-3.5 w-3.5" />} {copied ? 'Đã chép' : 'Sao chép'}
            </button>
          </div>

          <AnswerFeedback response={response} compact />
          <PipelineDisclosure response={response} showDebug={showDebug} />
        </div>

        {onFollowUpSelect ? (
          <div className="mt-3">
            <FollowUpInsights
              followUps={followUps}
              loadState={followUpsState}
              latencyMs={followUpsLatencyMs}
              onSelect={onFollowUpSelect}
              onSpeak={onFollowUpSpeak}
              showDebug={showDebug}
            />
          </div>
        ) : null}
      </div>
    </article>
  );
}
