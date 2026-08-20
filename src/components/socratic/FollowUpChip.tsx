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
    <div className="flex max-w-full items-stretch overflow-hidden rounded-xl border border-violet-200 bg-violet-50 text-violet-800 transition hover:border-violet-300 hover:bg-violet-100">
      <button type="button" onClick={() => onSelect(followUp)} aria-label={`Hỏi tiếp: ${followUp.question}`} className="flex min-w-0 flex-1 items-center gap-2 px-3 py-2 text-left text-sm leading-5">
        <ArrowUpRight className="h-4 w-4 shrink-0" />
        <span>{followUp.question}</span>
      </button>
      {onSpeak ? (
        <button type="button" onClick={() => onSpeak(followUp)} aria-label={`Đọc câu gợi ý: ${followUp.question}`} className="m-1 inline-flex w-8 shrink-0 items-center justify-center rounded-lg hover:bg-white/60">
          <Volume2 className="h-3.5 w-3.5" />
        </button>
      ) : null}
      {showDebug ? <span className="sr-only">Relation {followUp.relation ?? '--'}, ranking {followUp.ranking_score.toFixed(2)}</span> : null}
    </div>
  );
}
