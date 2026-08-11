import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Clock3, Cpu, Database, MessageSquareText, ShieldAlert, ChevronDown, ChevronUp } from 'lucide-react';
import type { PipelineState } from '@/types/pipeline';
import type { PassageResult, QaResponse } from '@/types/qa';
import { formatLatency } from '@/utils/formatScore';
import { ConfidenceBadge } from './ConfidenceBadge';
import { SourceCard } from './SourceCard';
import { PassageCard } from './PassageCard';

interface AnswerPanelProps {
  response: QaResponse | null;
  state: PipelineState;
  compareMode: boolean;
  onViewSource?: (passage: PassageResult) => void;
}

export function AnswerPanel({ response, state, compareMode, onViewSource }: AnswerPanelProps) {
  const [isPassagesCollapsed, setIsPassagesCollapsed] = useState(false);
  const hasAnswer = response?.has_answer ?? Boolean(response?.answer);
  const lowConfidence = Boolean(response && hasAnswer && response.confidence < 0.5);
  const noAnswerWithRetrievedPassage = Boolean(response && !hasAnswer && response.passages?.length);

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
          <p className="text-sm text-slate-400">{response?.question || 'Ask Mari a question about your documents.'}</p>
        </div>
      </div>

      <div className={`rounded-lg border p-5 ${hasAnswer ? 'border-viqa-gold/25 bg-amber-400/[0.08]' : 'border-slate-400/15 bg-[#172033]'}`}>
        {response?.answer ? (
          <p className="text-lg font-medium leading-8 text-slate-50">
            <span className="text-amber-200">{response.answer}</span>
          </p>
        ) : (
          <p className="text-base leading-7 text-slate-300">
            Không tìm thấy câu trả lời đủ tin cậy trong tập tài liệu.
          </p>
        )}
        {lowConfidence ? (
          <div className="mt-4 flex items-start gap-2 rounded-lg border border-amber-400/20 bg-amber-400/10 px-3 py-2.5 text-sm text-amber-100">
            <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
            Câu trả lời có độ tin cậy thấp. Vui lòng kiểm tra nguồn.
          </div>
        ) : null}
      </div>

      {response?.answer_source ? <SourceCard source={response.answer_source} /> : null}

      {noAnswerWithRetrievedPassage ? (
        <div className="rounded-lg border border-amber-400/20 bg-amber-400/10 p-4 text-sm leading-6 text-amber-50">
          <p className="font-medium">No answer debug</p>
          <p className="mt-1 text-amber-100/90">{response?.no_answer_reason ?? 'Reader did not return an answer above threshold.'}</p>
          <p className="text-amber-100/80">
            Best reader score: {typeof response?.best_reader_score === 'number' ? `${(response.best_reader_score * 100).toFixed(1)}%` : '--'}
            {response?.scoring ? ` | Threshold: ${(response.scoring.answer_threshold * 100).toFixed(1)}%` : ''}
          </p>
          <p className="text-amber-100/70">Most relevant retrieved passage is shown below, but it is not treated as an answer source.</p>
        </div>
      ) : null}

      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-y border-slate-400/15 py-3 text-xs text-slate-400">
        {response ? <ConfidenceBadge confidence={response.confidence} /> : <span>Confidence --</span>}
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
                title={isPassagesCollapsed ? 'Mở rộng danh sách nguồn' : 'Thu gọn danh sách nguồn'}
              >
                {isPassagesCollapsed ? (
                  <>
                    <ChevronDown className="h-3.5 w-3.5" />
                    <span>Mở rộng</span>
                  </>
                ) : (
                  <>
                    <ChevronUp className="h-3.5 w-3.5" />
                    <span>Thu gọn</span>
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
