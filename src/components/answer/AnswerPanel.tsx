import { motion } from 'framer-motion';
import { Clock3, ShieldAlert, Sparkles } from 'lucide-react';
import type { PipelineState } from '@/types/pipeline';
import type { PassageResult, QaResponse } from '@/types/qa';
import { formatLatency, formatScore } from '@/utils/formatScore';
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
  const hasAnswer = Boolean(response?.answer);
  const lowConfidence = Boolean(response && response.confidence < 0.5);

  return (
    <motion.section initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} className="viqa-panel flex h-full min-h-[580px] flex-col gap-4 rounded-[30px] p-4 lg:p-5">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-slate-500">Answer Intelligence</p>
          <h2 className="mt-1 font-display text-xl tracking-[0.14em] text-slate-50">ANSWER</h2>
        </div>
        {response ? <ConfidenceBadge confidence={response.confidence} /> : null}
      </div>

      <div className="grid gap-3 rounded-[24px] border border-white/10 bg-black/20 p-4 md:grid-cols-2">
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Status</p>
          <p className="text-sm text-slate-200">{state.toUpperCase()}</p>
          <div className="text-xs text-slate-400">{response ? response.question : 'Chưa có câu hỏi nào được xử lý.'}</div>
        </div>
        <div className="grid gap-2 text-sm text-slate-300">
          <div className="flex items-center gap-2">
            <Clock3 className="h-4 w-4 text-viqa-cyan" />
            Response time: {response ? formatLatency(response.processing_time_ms) : '--'}
          </div>
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-viqa-gold" />
            Retriever: {response ? response.retriever.toUpperCase() : '--'}
          </div>
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-viqa-violet" />
            Reader: {response ? response.reader.toUpperCase() : '--'}
          </div>
        </div>
      </div>

      <div className={`rounded-[28px] border p-5 ${hasAnswer ? 'border-viqa-gold/25 bg-viqa-gold/10' : 'border-white/10 bg-white/5'}`}>
        <div className="flex items-center gap-2 text-xs uppercase tracking-[0.32em] text-slate-500">
          <ShieldAlert className={`h-4 w-4 ${lowConfidence ? 'text-amber-300' : 'text-viqa-gold'}`} />
          Current answer
        </div>
        {response?.answer ? (
          <p className="mt-4 text-[17px] leading-8 text-slate-50">
            <span className="font-semibold text-viqa-gold">{response.answer}</span>
          </p>
        ) : (
          <p className="mt-4 text-[17px] leading-8 text-slate-300">
            Hệ thống chưa tìm thấy câu trả lời đáng tin cậy trong tập tài liệu.
          </p>
        )}
        {lowConfidence ? (
          <p className="mt-4 rounded-2xl border border-amber-400/20 bg-amber-400/10 px-4 py-3 text-sm text-amber-100">
            Câu trả lời có độ tin cậy thấp. Vui lòng kiểm tra nguồn.
          </p>
        ) : null}
      </div>

      {response ? <SourceCard source={response.source} /> : null}

      <div className="flex-1 rounded-[28px] border border-white/10 bg-white/5 p-4">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Top-k passages</p>
            <h3 className="mt-1 text-lg font-semibold text-white">Source passages</h3>
          </div>
          {compareMode ? <span className="rounded-full border border-viqa-cyan/20 bg-viqa-cyan/10 px-3 py-1 text-xs text-viqa-cyan">Compare mode</span> : null}
        </div>

        <div className="grid gap-3">
          {response?.passages?.length ? (
            response.passages.map((passage) => (
              <PassageCard
                key={passage.passage_id}
                passage={passage}
                answer={response.answer || response.answer_span?.text}
                highlighted={passage.rank === 1}
                onViewSource={onViewSource}
              />
            ))
          ) : (
            <div className="rounded-[24px] border border-dashed border-white/10 px-4 py-8 text-center text-sm text-slate-400">
              Top-k passages sẽ xuất hiện ở đây sau khi truy xuất tài liệu.
            </div>
          )}
        </div>
      </div>
    </motion.section>
  );
}