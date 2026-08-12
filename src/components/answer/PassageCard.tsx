import { useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowUpRight, ChevronDown, ChevronUp } from 'lucide-react';
import type { PassageResult } from '@/types/qa';
import { highlightAnswer } from '@/utils/highlightAnswer';

interface PassageCardProps {
  passage: PassageResult;
  answer?: string | null;
  highlighted?: boolean;
  onViewSource?: (passage: PassageResult) => void;
}

const showDebugScores = import.meta.env.DEV || String(import.meta.env.VITE_QA_DEBUG ?? 'false').toLowerCase() === 'true';

export function PassageCard({ passage, answer, highlighted = false, onViewSource }: PassageCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const isLongText = passage.text.length > 220;
  const displayedText = isLongText && !isExpanded ? `${passage.text.slice(0, 220)}...` : passage.text;

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
            {highlighted ? <span className="rounded-full bg-viqa-cyan/10 px-2 py-1 font-medium text-viqa-cyan">Selected answer source</span> : null}
            <span className="font-medium text-slate-300">{passage.title}</span>
          </div>
          <p className="mt-2 text-sm text-slate-300">{passage.passage_id} {passage.page ? `• Page ${passage.page}` : ''}</p>
        </div>
        <div className="text-right text-xs text-slate-400">
          <p title="Normalized within retrieved candidates; not a correctness probability.">
            Retrieval score {(passage.retrieval_score_normalized ?? passage.retrieval_score).toFixed(3)}
          </p>
          {typeof passage.reader_score === 'number' ? (
            <p title="Uncalibrated Reader/fallback ranking signal; not a probability.">Reader score {passage.reader_score.toFixed(3)}</p>
          ) : null}
          {typeof passage.answer_type_score === 'number' ? <p>Answer-type score {passage.answer_type_score.toFixed(3)}</p> : null}
          {typeof passage.ranking_score === 'number' ? (
            <p title="Used to order candidates; not a correctness probability.">Ranking score {passage.ranking_score.toFixed(3)}</p>
          ) : null}
        </div>
      </div>

      <p className="mt-4 text-sm leading-7 text-slate-200">{highlightAnswer(displayedText, answer ?? undefined)}</p>

      {showDebugScores ? (
        <div className="mt-3 border-t border-slate-400/15 pt-3 text-xs leading-6 text-slate-400">
          <p>Retriever raw: {passage.retrieval_score_raw?.toFixed(4) ?? '--'}</p>
          <p>Retriever normalized: {passage.retrieval_score_normalized?.toFixed(4) ?? '--'}</p>
          <p>Original retrieval rank: {passage.retrieval_rank ?? '--'}</p>
          <p>Question type: {passage.question_type ?? '--'}</p>
          <p>Reader method: {passage.reader_method ?? 'neural_span'}</p>
          <p>Reader candidate: {passage.reader_answer || 'No span'}</p>
          <p>Neural score: {passage.neural_reader_score?.toFixed(4) ?? '--'}</p>
          <p>Fallback score: {passage.fallback_score?.toFixed(4) ?? '--'}</p>
          <p>Reader margin: {passage.reader_score_margin?.toFixed(4) ?? '--'}</p>
          <p>Answer-type score: {passage.answer_type_score?.toFixed(4) ?? '--'} ({passage.answer_type_reason ?? '--'})</p>
          {passage.question_type === 'LOCATION' ? (
            <>
              <p>Relation: {passage.relation_type ?? '--'} ({passage.relation_score?.toFixed(4) ?? '0.0000'})</p>
              <p>Location phrase quality: {passage.phrase_quality?.toFixed(4) ?? '--'}</p>
              <p>Lexical evidence: {passage.lexical_evidence ? 'yes' : 'no'}</p>
              <p>Relation evidence: {passage.relation_evidence ? 'yes' : 'no'}</p>
            </>
          ) : null}
          <p>Evidence supported: {passage.evidence_supported ? 'yes' : 'no'}</p>
          <p>Ranking score: {passage.ranking_score?.toFixed(4) ?? '--'}</p>
          <p>Status: {passage.selection_status ?? 'REJECTED'}</p>
          {passage.rejection_reason ? <p>Rejected: {passage.rejection_reason} — {passage.rejection_detail}</p> : null}
        </div>
      ) : null}

      <div className="mt-4 flex items-center justify-between gap-2 border-t border-slate-400/10 pt-3">
        {isLongText ? (
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            className="inline-flex items-center gap-1 text-xs font-medium text-viqa-cyan transition hover:underline"
          >
            {isExpanded ? (
              <>
                <ChevronUp className="h-3.5 w-3.5" />
                <span>Thu gọn văn bản</span>
              </>
            ) : (
              <>
                <ChevronDown className="h-3.5 w-3.5" />
                <span>Xem thêm</span>
              </>
            )}
          </button>
        ) : (
          <div />
        )}

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
