import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Clock3, Cpu, Database, MessageSquareText, ChevronDown, ChevronUp, LoaderCircle } from 'lucide-react';
import type { PipelineState } from '@/types/pipeline';
import type { FollowUpCandidate, PassageResult, QaResponse } from '@/types/qa';
import type { SocraticLoadState } from '@/hooks/useSocraticFollowups';
import { formatLatency } from '@/utils/formatScore';
import { FollowUpInsights } from '@/components/socratic/FollowUpInsights';
import { AnswerFeedback } from '@/components/feedback/AnswerFeedback';
import { SourceCard } from './SourceCard';
import { PassageCard } from './PassageCard';

interface AnswerPanelProps {
  response: QaResponse | null;
  submittedQuestion?: string;
  state: PipelineState;
  compareMode: boolean;
  onViewSource?: (passage: PassageResult) => void;
  followUps: FollowUpCandidate[];
  followUpsState: SocraticLoadState;
  followUpsLatencyMs?: number | null;
  onFollowUpSelect: (followUp: FollowUpCandidate) => void;
  onFollowUpSpeak?: (followUp: FollowUpCandidate) => void;
  showSocraticDebug?: boolean;
}

export function AnswerPanel({
  response,
  submittedQuestion,
  state,
  compareMode,
  onViewSource,
  followUps,
  followUpsState,
  followUpsLatencyMs,
  onFollowUpSelect,
  onFollowUpSpeak,
  showSocraticDebug = false,
}: AnswerPanelProps) {
  const [isPassagesCollapsed, setIsPassagesCollapsed] = useState(false);
  const hasAnswer = response?.has_answer ?? Boolean(response?.answer);
  const noAnswerWithRetrievedPassage = Boolean(response && !hasAnswer && response.passages?.length);
  const isProcessing = !response && ['retrieving', 'reading', 'extracting'].includes(state);
  const isConfirmedNoAnswer = Boolean(response && !hasAnswer);
  const displayedQuestion = response?.question || submittedQuestion || 'Ask Mari a question about your documents.';

  const handleScrollToPassage = (passageId: string) => {
    if (isPassagesCollapsed) {
      setIsPassagesCollapsed(false);
      setTimeout(() => {
        document.getElementById(`passage-${passageId}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }, 150);
    } else {
      document.getElementById(`passage-${passageId}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  return (
    <motion.section
      initial={{ opacity: 0, x: 16 }}
      animate={{ opacity: 1, x: 0 }}
      className={`viqa-panel flex flex-col gap-4 p-4 lg:p-5 ${
        isPassagesCollapsed ? 'h-auto min-h-0' : 'h-auto min-h-[580px]'
      }`}
    >
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-viqa-cyan/10 text-viqa-cyan">
          <MessageSquareText className="h-5 w-5" />
        </div>
        <div>
          <h2 className="font-display text-xl font-semibold text-slate-50">Answer</h2>
          <p className="text-sm text-slate-400">{displayedQuestion}</p>
        </div>
      </div>

      <div className={`rounded-lg border p-5 ${hasAnswer ? 'border-viqa-gold/25 bg-amber-400/[0.08]' : 'border-slate-400/15 bg-[#172033]'}`}>
        {response?.answer ? (
          <p className="text-lg font-medium leading-8 text-slate-50">
            <span className="text-amber-200">{response.answer}</span>
          </p>
        ) : isProcessing ? (
          <div className="flex items-center gap-3 text-base leading-7 text-slate-300" role="status" aria-live="polite">
            <LoaderCircle className="h-5 w-5 shrink-0 animate-spin text-viqa-cyan" />
            <span>Thinking about the answer...</span>
          </div>
        ) : isConfirmedNoAnswer ? (
          <p className="text-base leading-7 text-slate-300">
            Không tìm thấy câu trả lời đủ tin cậy trong các đoạn được truy xuất.
          </p>
        ) : (
          <p className="text-base leading-7 text-slate-300">
            Ask a question to get started.
          </p>
        )}
      </div>

      {response?.answer_source ? <SourceCard source={response.answer_source} onScrollToPassage={handleScrollToPassage} /> : null}

      {response ? <AnswerFeedback response={response} /> : null}

      {response && hasAnswer ? (
        <FollowUpInsights
          followUps={followUps}
          loadState={followUpsState}
          latencyMs={followUpsLatencyMs}
          onSelect={onFollowUpSelect}
          onSpeak={onFollowUpSpeak}
          showDebug={showSocraticDebug}
        />
      ) : null}

      {noAnswerWithRetrievedPassage ? (
        <div className="rounded-lg border border-amber-400/20 bg-amber-400/10 p-4 text-sm leading-6 text-amber-50">
          <p className="font-medium">No answer debug</p>
          <p className="mt-1 text-amber-100/90">{response?.no_answer_reason ?? 'Reader did not return an answer above threshold.'}</p>
          <p className="text-amber-100/80">
            Rejection: {response?.rejection_reason ?? 'NO_ANSWER'}
            {response?.rejection_detail ? ` — ${response.rejection_detail}` : ''}
          </p>
          <p className="text-amber-100/70">
            Top passages were relevant to the topic, but no candidate matched the expected answer type with sufficient Reader evidence.
          </p>
          <p className="text-amber-100/70">The most relevant retrieved passage is shown below; it is not an answer source.</p>
        </div>
      ) : null}

      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-y border-slate-400/15 py-3 text-xs text-slate-400">
        <span>
          Answer confidence: {typeof response?.answer_confidence === 'number' ? `${(response.answer_confidence * 100).toFixed(1)}%` : 'not calibrated'}
        </span>
        <span>Question type: {Array.isArray(response?.question_type) ? response.question_type.join(', ') : (response?.question_type ?? '--')}</span>
        <span>Reader method: {response?.reader_method ?? '--'}</span>
        {response?.fallback_method ? <span>Fallback method: {response.fallback_method}</span> : null}
        {(Array.isArray(response?.question_type) ? response.question_type.includes('LOCATION') : response?.question_type === 'LOCATION') ? (
          <span>
            Relation: {response?.relation_type ?? '--'} ({response?.relation_score?.toFixed(2) ?? '0.00'})
          </span>
        ) : null}
        <span>Reader score: {response?.scores?.reader?.toFixed(3) ?? '--'}</span>
        <span title="Used for candidate ordering; not a correctness probability.">Ranking score: {response?.scores?.ranking?.toFixed(3) ?? '--'}</span>
        <span className="inline-flex items-center gap-1.5">
          <Clock3 className="h-3.5 w-3.5 text-viqa-cyan" />
          {response ? formatLatency(response.processing_time_ms) : '--'}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <Database className="h-3.5 w-3.5 text-viqa-cyan" />
          Retriever: {response ? response.retriever.toUpperCase() : '--'}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <Cpu className="h-3.5 w-3.5 text-viqa-violet" />
          Reader: {response ? response.reader.toUpperCase() : '--'}
        </span>
        <span className="capitalize text-slate-500">Status: {state}</span>
      </div>

      <div className={isPassagesCollapsed ? 'flex-none' : 'flex-1'}>
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <h3 className="font-display text-base font-semibold text-slate-100">Source passages</h3>
            {response?.passages?.length ? (
              <span className="rounded-full bg-slate-800 px-2.5 py-0.5 text-xs font-medium text-slate-400">
                {response.passages.length}
              </span>
            ) : null}
          </div>
          <div className="flex items-center gap-2">
            {compareMode ? <span className="rounded-full border border-viqa-cyan/20 bg-viqa-cyan/10 px-3 py-1 text-xs text-viqa-cyan">Compare mode</span> : null}
            {response?.passages?.length ? (
              <button
                type="button"
                onClick={() => setIsPassagesCollapsed(!isPassagesCollapsed)}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-400/15 bg-slate-800/80 px-2.5 py-1 text-xs font-medium text-slate-300 transition hover:border-viqa-cyan/30 hover:text-viqa-cyan"
                title={isPassagesCollapsed ? 'Expand source passages' : 'Collapse source passages'}
              >
                {isPassagesCollapsed ? (
                  <>
                    <ChevronDown className="h-3.5 w-3.5" />
                    <span>Expand</span>
                  </>
                ) : (
                  <>
                    <ChevronUp className="h-3.5 w-3.5" />
                    <span>Collapse</span>
                  </>
                )}
              </button>
            ) : null}
          </div>
        </div>

        <AnimatePresence initial={false}>
          {!isPassagesCollapsed && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.25 }}
              className="mt-3 grid gap-3 overflow-hidden"
            >
              {response?.passages?.length ? (
                response.passages.map((passage) => (
                  <PassageCard
                    key={passage.passage_id}
                    passage={passage}
                    answer={response.answer || response.answer_span?.text}
                    highlighted={passage.passage_id === response.selected_passage_id}
                    onViewSource={onViewSource}
                  />
                ))
              ) : (
                <div className="rounded-lg border border-dashed border-slate-400/15 bg-slate-800/35 px-4 py-8 text-center text-sm text-slate-400">
                  Retrieved passages will appear here.
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.section>
  );
}
