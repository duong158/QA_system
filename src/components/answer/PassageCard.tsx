import { motion } from 'framer-motion';
import { ArrowUpRight } from 'lucide-react';
import type { PassageResult } from '@/types/qa';
import { formatScore } from '@/utils/formatScore';
import { highlightAnswer } from '@/utils/highlightAnswer';

interface PassageCardProps {
  passage: PassageResult;
  answer?: string;
  highlighted?: boolean;
  onViewSource?: (passage: PassageResult) => void;
}

export function PassageCard({ passage, answer, highlighted = false, onViewSource }: PassageCardProps) {
  return (
    <motion.article
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: passage.rank * 0.06 }}
      className={`rounded-[24px] border p-4 ${highlighted ? 'border-viqa-gold/25 bg-viqa-gold/10' : 'border-white/10 bg-white/5'}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-xs uppercase tracking-[0.3em] text-slate-500">
            <span className={`rounded-full px-2 py-1 ${highlighted ? 'bg-viqa-gold/15 text-viqa-gold' : 'bg-white/5 text-slate-300'}`}>Rank {passage.rank}</span>
            <span>{passage.title}</span>
          </div>
          <p className="mt-2 text-sm text-slate-300">{passage.passage_id} {passage.page ? `• Page ${passage.page}` : ''}</p>
        </div>
        <div className="text-right text-xs text-slate-400">
          <p>Retrieval {formatScore(passage.retrieval_score)}</p>
          {typeof passage.reader_score === 'number' ? <p>Reader {formatScore(passage.reader_score)}</p> : null}
        </div>
      </div>

      <p className="mt-4 text-sm leading-7 text-slate-200">{highlightAnswer(passage.text, answer)}</p>

      <div className="mt-4 flex items-center justify-end">
        <button
          type="button"
          onClick={() => onViewSource?.(passage)}
          className="inline-flex items-center gap-1 rounded-full border border-white/10 bg-black/20 px-3 py-1.5 text-xs text-slate-300 transition hover:border-viqa-cyan/30 hover:text-viqa-cyan"
        >
          View source
          <ArrowUpRight className="h-3.5 w-3.5" />
        </button>
      </div>
    </motion.article>
  );
}