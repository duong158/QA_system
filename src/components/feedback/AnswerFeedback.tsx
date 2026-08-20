import { useEffect, useMemo, useRef, useState } from 'react';
import { Check, MessageSquareWarning, MousePointer2, ThumbsDown, ThumbsUp, X } from 'lucide-react';
import { submitFeedback } from '@/services/feedbackService';
import type { FeedbackSubmitRequest } from '@/types/feedback';
import type { PassageResult, QaResponse } from '@/types/qa';
import { getSelectionOffsets, type TextSelectionOffsets } from '@/utils/textSelection';

interface AnswerFeedbackProps {
  response: QaResponse;
  compact?: boolean;
}

type FeedbackMode = 'idle' | 'options' | 'span' | 'note';

export function AnswerFeedback({ response, compact = false }: AnswerFeedbackProps) {
  const [mode, setMode] = useState<FeedbackMode>('idle');
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState('');
  const [selectedPassageId, setSelectedPassageId] = useState('');
  const [span, setSpan] = useState<TextSelectionOffsets | null>(null);
  const sourceRef = useRef<HTMLParagraphElement>(null);
  const hasAnswer = response.has_answer ?? Boolean(response.answer);

  const passages = response.passages ?? [];
  const selectedPassage = useMemo<PassageResult | null>(
    () => passages.find((item) => item.passage_id === selectedPassageId) ?? null,
    [passages, selectedPassageId],
  );

  useEffect(() => {
    setMode('idle');
    setSubmitting(false);
    setMessage(null);
    setError(null);
    setNote('');
    setSpan(null);
    setSelectedPassageId(
      response.selected_passage_id
      || response.passages?.[0]?.passage_id
      || '',
    );
  }, [response]);

  const basePayload = (): Omit<FeedbackSubmitRequest, 'feedback_type'> => ({
    question: response.question,
    predicted_answer: response.answer,
    question_type: Array.isArray(response.question_type)
      ? response.question_type[0] ?? null
      : response.question_type ?? null,
    semantic_relation: response.semantic_relation ?? response.question_relation ?? response.relation_type ?? null,
    subject: response.question_subject ?? null,
    selected_passage_id: response.selected_passage_id ?? null,
    retrieved_passage_ids: Array.from(new Set(passages.map((item) => item.passage_id))),
    rejection_reason: response.rejection_reason ?? null,
    source: 'DIRECT_QUERY',
  });

  const send = async (payload: FeedbackSubmitRequest) => {
    setSubmitting(true);
    setError(null);
    try {
      const result = await submitFeedback(payload);
      setMessage(result.message);
      setMode('idle');
      window.getSelection()?.removeAllRanges();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Không thể gửi phản hồi.');
    } finally {
      setSubmitting(false);
    }
  };

  const markCorrect = () => {
    void send({ ...basePayload(), feedback_type: 'CORRECT' });
  };

  const markShouldNotAnswer = () => {
    void send({ ...basePayload(), feedback_type: 'ANSWERED_BUT_SHOULD_NOT' });
  };

  const submitNote = () => {
    if (!note.trim()) {
      return;
    }
    void send({
      ...basePayload(),
      feedback_type: hasAnswer ? 'INCORRECT' : 'NO_ANSWER_BUT_SHOULD_HAVE',
      user_note: note.trim(),
    });
  };

  const captureSelection = () => {
    if (!sourceRef.current) {
      setSpan(null);
      return;
    }
    setSpan(getSelectionOffsets(sourceRef.current, window.getSelection()));
  };

  const submitSpan = () => {
    if (!selectedPassage || !span) {
      return;
    }
    void send({
      ...basePayload(),
      feedback_type: hasAnswer ? 'SPAN_CORRECTION' : 'NO_ANSWER_BUT_SHOULD_HAVE',
      corrected_passage_id: selectedPassage.passage_id,
      corrected_answer: span.text,
      corrected_start_char: span.start,
      corrected_end_char: span.end,
    });
  };

  if (message) {
    return (
      <div className="mt-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-700" role="status">
        <span className="inline-flex items-start gap-2">
          <Check className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
          {message}
        </span>
        <p className="mt-1 pl-6 text-xs text-emerald-600/75">Phản hồi được đưa vào hàng chờ kiểm duyệt; hệ thống không tự cập nhật trọng số hay đổi model tại runtime.</p>
      </div>
    );
  }

  return (
    <section className={compact ? 'mt-1 border-t border-[var(--border)] pt-2' : 'rounded-lg border border-[var(--border)] bg-[var(--surface-subtle)] p-3'} aria-label="Phản hồi câu trả lời">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-xs text-[var(--text-muted)]">Câu trả lời này có hữu ích không?</p>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={markCorrect}
            disabled={submitting}
            className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs text-[var(--text-secondary)] transition hover:bg-emerald-50 hover:text-emerald-700 disabled:opacity-50"
          >
            <ThumbsUp className="h-3.5 w-3.5" />
            {hasAnswer ? 'Chính xác' : 'Đúng, tài liệu không có'}
          </button>
          <button
            type="button"
            onClick={() => setMode(mode === 'options' ? 'idle' : 'options')}
            disabled={submitting}
            className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs text-[var(--text-secondary)] transition hover:bg-rose-50 hover:text-rose-700 disabled:opacity-50"
          >
            <ThumbsDown className="h-3.5 w-3.5" />
            {hasAnswer ? 'Chưa chính xác' : 'Sai, tài liệu có câu trả lời'}
          </button>
        </div>
      </div>

      {mode === 'options' ? (
        <div className="mt-3 border-t border-[var(--border)] pt-3">
          <p className="mb-2 text-sm font-medium text-[var(--text-primary)]">Câu trả lời sai ở đâu?</p>
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={() => setMode('span')} className="rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2 text-xs text-indigo-700">
              Chọn đoạn đúng trong nguồn
            </button>
            {hasAnswer ? (
              <button type="button" onClick={markShouldNotAnswer} className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
                Đáng lẽ hệ thống không nên trả lời
              </button>
            ) : null}
            <button type="button" onClick={() => setMode('note')} className="rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] px-3 py-2 text-xs text-[var(--text-secondary)]">
              Gửi ghi chú
            </button>
          </div>
        </div>
      ) : null}

      {mode === 'span' ? (
        <div className="mt-3 border-t border-[var(--border)] pt-3">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="inline-flex items-center gap-2 text-sm font-medium text-[var(--text-primary)]">
                <MousePointer2 className="h-4 w-4 text-indigo-600" />
                Bôi đen đáp án đúng trong passage
              </p>
              <p className="mt-1 text-xs text-[var(--text-secondary)]">Chỉ đoạn văn thực trong corpus mới được chấp nhận.</p>
            </div>
            <button type="button" onClick={() => setMode('options')} aria-label="Đóng chọn đoạn"><X className="h-4 w-4 text-[var(--text-secondary)]" /></button>
          </div>

          <select
            value={selectedPassageId}
            onChange={(event) => { setSelectedPassageId(event.target.value); setSpan(null); }}
            className="mt-3 w-full rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-xs text-[var(--text-primary)] outline-none"
            aria-label="Chọn passage nguồn"
          >
            {passages.map((passage) => (
              <option key={passage.passage_id} value={passage.passage_id}>
                Rank {passage.rank} · {passage.passage_id}
              </option>
            ))}
          </select>

          {selectedPassage ? (
            <p
              ref={sourceRef}
              onMouseUp={captureSelection}
              onKeyUp={captureSelection}
              className="mt-3 max-h-52 select-text overflow-y-auto rounded-lg border border-[var(--border)] bg-[var(--surface-subtle)] p-3 text-sm leading-7 text-[var(--text-primary)] selection:bg-indigo-200"
              tabIndex={0}
            >
              {selectedPassage.text}
            </p>
          ) : (
            <p className="mt-3 text-sm text-amber-700">Không có passage để chọn đoạn.</p>
          )}

          <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
            <p className="min-w-0 flex-1 truncate text-xs text-[var(--text-secondary)]">
              {span ? `Đã chọn [${span.start}:${span.end}]: “${span.text}”` : 'Hãy bôi đen một đoạn trước khi gửi.'}
            </p>
            <button
              type="button"
              onClick={submitSpan}
              disabled={!span || submitting}
              className="rounded-lg bg-[var(--primary)] px-3 py-2 text-xs font-semibold text-white disabled:cursor-not-allowed disabled:opacity-40"
            >
              Gửi đoạn sửa
            </button>
          </div>
        </div>
      ) : null}

      {mode === 'note' ? (
        <div className="mt-3 border-t border-[var(--border)] pt-3">
          <label className="text-sm font-medium text-[var(--text-primary)]" htmlFor="feedback-note">Ghi chú cho người kiểm duyệt</label>
          <textarea
            id="feedback-note"
            value={note}
            onChange={(event) => setNote(event.target.value)}
            rows={3}
            placeholder="Mô tả ngắn vấn đề..."
            className="mt-2 w-full rounded-lg border border-[var(--border)] bg-[var(--surface-subtle)] px-3 py-2 text-sm text-[var(--text-primary)] outline-none focus:border-indigo-300"
          />
          <button
            type="button"
            onClick={submitNote}
            disabled={!note.trim() || submitting}
            className="mt-2 inline-flex items-center gap-2 rounded-lg bg-[var(--primary)] px-3 py-2 text-xs font-semibold text-white disabled:opacity-40"
          >
            <MessageSquareWarning className="h-3.5 w-3.5" />
            Gửi ghi chú
          </button>
        </div>
      ) : null}

      {error ? <p className="mt-3 text-xs text-rose-700" role="alert">{error}</p> : null}
    </section>
  );
}
