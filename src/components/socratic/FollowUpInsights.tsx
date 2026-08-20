import { LoaderCircle, Sparkles } from 'lucide-react';
import type { FollowUpCandidate } from '@/types/qa';
import type { SocraticLoadState } from '@/hooks/useSocraticFollowups';
import { FollowUpChip } from './FollowUpChip';

interface FollowUpInsightsProps {
  followUps: FollowUpCandidate[];
  loadState: SocraticLoadState;
  latencyMs?: number | null;
  onSelect: (followUp: FollowUpCandidate) => void;
  onSpeak?: (followUp: FollowUpCandidate) => void;
  showDebug?: boolean;
}

export function FollowUpInsights({
  followUps,
  loadState,
  latencyMs,
  onSelect,
  onSpeak,
  showDebug = false,
}: FollowUpInsightsProps) {
  if (loadState === 'idle' || loadState === 'error') {
    return null;
  }

  return (
    <section className="rounded-xl border border-viqa-violet/20 bg-viqa-violet/[0.06] p-3" aria-live="polite">
      <div className="mb-2.5 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm font-medium text-slate-100">
          <Sparkles className="h-4 w-4 text-viqa-violet" />
          Khám phá thêm
        </div>
        {showDebug && typeof latencyMs === 'number' ? (
          <span className="text-[10px] text-slate-500">{latencyMs} ms</span>
        ) : null}
      </div>

      {loadState === 'loading' ? (
        <div className="flex items-center gap-2 py-1 text-sm text-slate-400" role="status">
          <LoaderCircle className="h-4 w-4 animate-spin text-viqa-cyan" />
          Đang tìm hướng khám phá tiếp từ tài liệu...
        </div>
      ) : followUps.length ? (
        <div className="grid gap-2">
          {followUps.map((followUp) => (
            <FollowUpChip
              key={`${followUp.relation ?? 'GENERAL'}-${followUp.question}`}
              followUp={followUp}
              onSelect={onSelect}
              onSpeak={onSpeak}
              showDebug={showDebug}
            />
          ))}
        </div>
      ) : (
        <p className="text-sm leading-5 text-slate-400">
          Chưa có gợi ý khám phá phù hợp từ tài liệu hiện tại.
        </p>
      )}
    </section>
  );
}
