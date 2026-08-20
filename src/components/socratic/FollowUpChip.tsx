import { ArrowUpRight, Volume2 } from 'lucide-react';
import type { FollowUpCandidate } from '@/types/qa';

interface FollowUpChipProps {
  followUp: FollowUpCandidate;
  onSelect: (followUp: FollowUpCandidate) => void;
  onSpeak?: (followUp: FollowUpCandidate) => void;
  showDebug?: boolean;
}

export function FollowUpChip({ followUp, onSelect, onSpeak, showDebug = false }: FollowUpChipProps) {
  return (
    <div className="group rounded-xl border border-slate-400/15 bg-slate-800/55 transition hover:border-viqa-cyan/35 hover:bg-slate-800/80">
      <div className="flex items-stretch">
        <button
          type="button"
          onClick={() => onSelect(followUp)}
          aria-label={`Hỏi tiếp: ${followUp.question}`}
          className="flex min-w-0 flex-1 items-center gap-2 px-3 py-2.5 text-left text-sm leading-5 text-slate-200 outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-viqa-cyan/70"
        >
          <ArrowUpRight className="h-4 w-4 shrink-0 text-viqa-cyan transition group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
          <span>{followUp.question}</span>
        </button>
        {onSpeak ? (
          <button
            type="button"
            onClick={() => onSpeak(followUp)}
            aria-label={`Đọc câu gợi ý: ${followUp.question}`}
            title="Đọc câu gợi ý"
            className="m-1.5 inline-flex w-9 shrink-0 items-center justify-center rounded-lg text-slate-400 transition hover:bg-white/5 hover:text-viqa-gold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-viqa-gold/70"
          >
            <Volume2 className="h-4 w-4" />
          </button>
        ) : null}
      </div>
      {showDebug ? (
        <div className="border-t border-slate-400/10 px-3 py-2 text-[10px] leading-4 text-slate-500">
          Relation: {followUp.relation ?? '--'} · Source: {followUp.source_passage_id ?? '--'} · Answerability:{' '}
          {followUp.answerability_score.toFixed(2)} · Novelty: {followUp.novelty_score.toFixed(2)} · Ranking:{' '}
          {followUp.ranking_score.toFixed(2)}
        </div>
      ) : null}
    </div>
  );
}
