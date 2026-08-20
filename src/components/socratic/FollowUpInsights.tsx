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

export function FollowUpInsights({ followUps, loadState, latencyMs, onSelect, onSpeak, showDebug = false }: FollowUpInsightsProps) {
  if (loadState === 'idle' || loadState === 'error') return null;

  return (
    <section aria-live="polite">
      <div className="mb-2 flex items-center gap-2 text-xs font-medium text-violet-700">
        <Sparkles className="h-3.5 w-3.5" />
        Mari gợi ý bạn khám phá tiếp
        {showDebug && typeof latencyMs === 'number' ? <span className="ml-auto text-[10px] text-[var(--text-muted)]">{latencyMs} ms</span> : null}
      </div>
      {loadState === 'loading' ? (
        <div className="flex items-center gap-2 py-1 text-sm text-[var(--text-secondary)]" role="status">
          <LoaderCircle className="h-4 w-4 animate-spin text-violet-500" /> Đang tạo câu hỏi gợi mở...
        </div>
      ) : followUps.length ? (
        <div className="flex flex-wrap gap-2">
          {followUps.map((followUp) => (
            <FollowUpChip key={`${followUp.relation ?? 'GENERAL'}-${followUp.question}`} followUp={followUp} onSelect={onSelect} onSpeak={onSpeak} showDebug={showDebug} />
          ))}
        </div>
      ) : (
        <p className="text-sm text-[var(--text-secondary)]">Chưa có gợi ý phù hợp từ tài liệu hiện tại.</p>
      )}
    </section>
  );
}
