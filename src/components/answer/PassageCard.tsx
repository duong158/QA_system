import { motion } from 'framer-motion';
import { ArrowUpRight } from 'lucide-react';
import type { PassageResult } from '@/types/qa';
import { formatScore } from '@/utils/formatScore';
import { highlightAnswer } from '@/utils/highlightAnswer';

interface PassageCardProps {
  passage: PassageResult;
  answer?: string | null;
  highlighted?: boolean;
  onViewSource?: (passage: PassageResult) => void;
}

const showDebugScores = import.meta.env.DEV || String(import.meta.env.VITE_QA_DEBUG ?? 'false').toLowerCase() === 'true';

export function PassageCard({ passage, answer, highlighted = false, onViewSource }: PassageCardProps) {
  return (
    <motion.article
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: passage.rank * 0.06 }}
      className={`rounded-lg border p-4 ${highlighted ? 'border-viqa-cyan/35 bg-[#1C2A40]' : 'border-slate-400/15 bg-[#172033]'}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
            <span className={`rounded-full px-2 py-1 ${highlighted ? 'bg-viqa-cyan/10 text-viqa-cyan' : 'bg-slate-700/50 text-slate-300'}`}>Rank {passage.rank}</span>
            {highlighted ? <span className="rounded-full bg-viqa-cyan/10 px-2 py-1 font-medium text-viqa-cyan">Best match</span> : null}
            <span className="font-medium text-slate-300">{passage.title}</span>
          </div>
          <p className="mt-2 text-sm text-slate-300">{passage.passage_id} {passage.page ? `• Page ${passage.page}` : ''}</p>
        </div>
        <div className="text-right text-xs text-slate-400">
          <p>Retriever {formatScore(passage.retrieval_score_normalized ?? passage.retrieval_score)}</p>
          {typeof passage.reader_score === 'number' ? <p>Reader {formatScore(passage.reader_score)}</p> : null}
          {typeof passage.final_score === 'number' ? <p>Final {formatScore(passage.final_score)}</p> : null}
        </div>
      </div>

      <p className="mt-4 text-sm leading-7 text-slate-200">{highlightAnswer(passage.text, answer ?? undefined)}</p>

      {showDebugScores ? (
        <div className="mt-3 border-t border-slate-400/15 pt-3 text-xs leading-6 text-slate-400">
          <p>Retriever raw: {passage.retrieval_score_raw?.toFixed(4) ?? '--'}</p>
          <p>Retriever normalized: {passage.retrieval_score_normalized?.toFixed(4) ?? '--'}</p>
          <p>Reader candidate: {passage.reader_answer || 'No span'}</p>
          <p>Reader margin: {passage.reader_score_margin?.toFixed(4) ?? '--'}</p>
        </div>
      ) : null}

      <div className="mt-4 flex items-center justify-end">
        <button
          type="button"
          onClick={() => onViewSource?.(passage)}
          className="inline-flex items-center gap-1 rounded-lg border border-slate-400/15 bg-slate-700/40 px-3 py-1.5 text-xs text-slate-300 transition hover:border-viqa-cyan/30 hover:text-viqa-cyan"
        >
          View source
          <ArrowUpRight className="h-3.5 w-3.5" />
        </button>
      </div>
    </motion.article>
  );
}
