import { motion } from 'framer-motion';
import { CheckCircle2, Clock3, Target } from 'lucide-react';
import type { RetrieverComparisonRow } from '@/types/qa';
import { formatLatency, formatScore } from '@/utils/formatScore';

interface RetrieverComparisonProps {
  rows: RetrieverComparisonRow[];
}

export function RetrieverComparison({ rows }: RetrieverComparisonProps) {
  return (
    <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="viqa-panel rounded-[28px] p-4">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-slate-500">Compare mode</p>
          <h3 className="mt-1 font-display text-lg tracking-[0.16em] text-white">COMPARE RETRIEVERS</h3>
        </div>
        <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">TF-IDF | BM25 | DENSE</span>
      </div>

      <div className="grid gap-3 xl:grid-cols-3">
        {rows.map((row) => (
          <div key={row.retriever} className="rounded-[24px] border border-white/10 bg-black/20 p-4">
            <div className="flex items-center justify-between gap-2">
              <div>
                <p className="text-sm font-semibold text-white">{row.label}</p>
                <p className="text-xs text-slate-400">Correct passage rank: {row.correctPassageRank}</p>
              </div>
              <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-viqa-cyan/20 bg-viqa-cyan/10 text-viqa-cyan">
                <Target className="h-4 w-4" />
              </div>
            </div>

            <div className="mt-4 grid gap-2 text-sm text-slate-300">
              <div className="flex items-center justify-between">
                <span>Retrieval score</span>
                <span>{formatScore(row.retrievalScore)}</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Response time</span>
                <span>{formatLatency(row.responseTimeMs)}</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Recall@1</span>
                <span className={row.recallAt1 ? 'text-emerald-300' : 'text-rose-300'}>{row.recallAt1 ? 'Passed' : 'Failed'}</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Recall@3</span>
                <span className={row.recallAt3 ? 'text-emerald-300' : 'text-rose-300'}>{row.recallAt3 ? 'Passed' : 'Failed'}</span>
              </div>
            </div>

            <div className="mt-4 rounded-2xl border border-white/10 bg-white/5 p-3 text-xs leading-6 text-slate-300">
              <div className="mb-2 flex items-center gap-2 text-slate-400">
                <CheckCircle2 className="h-4 w-4 text-viqa-gold" />
                Passage preview
              </div>
              <p>{row.topPassagePreview}</p>
              <div className="mt-3 flex items-center gap-2 text-slate-500">
                <Clock3 className="h-3.5 w-3.5" />
                {row.retriever.toUpperCase()}
              </div>
            </div>
          </div>
        ))}
      </div>
    </motion.section>
  );
}