import { LoaderCircle, Sparkles } from 'lucide-react';
import type { FollowUpCandidate, SocraticDebugInfo } from '@/types/qa';
import type { SocraticLoadState } from '@/hooks/useSocraticFollowups';
import { FollowUpChip } from './FollowUpChip';

interface FollowUpInsightsProps {
  followUps: FollowUpCandidate[];
  loadState: SocraticLoadState;
  latencyMs?: number | null;
  onSelect: (followUp: FollowUpCandidate) => void;
  onSpeak?: (followUp: FollowUpCandidate) => void;
  showDebug?: boolean;
  debug?: SocraticDebugInfo | null;
}

export function FollowUpInsights({ followUps, loadState, latencyMs, onSelect, onSpeak, showDebug = false, debug }: FollowUpInsightsProps) {
  if (loadState === 'idle' || loadState === 'error') return null;
  if (loadState === 'ready' && followUps.length === 0 && !showDebug) return null;

  return (
    <section aria-live="polite">
      <div className="mb-2 flex items-center gap-2 text-xs font-medium text-violet-700">
        <Sparkles className="h-3.5 w-3.5" />
        Mari gợi ý bạn khám phá tiếp
        {showDebug && typeof latencyMs === 'number' ? <span className="ml-auto text-[10px] text-[var(--text-muted)]">{latencyMs} ms</span> : null}
      </div>
      {loadState === 'loading' ? (
        <div className="flex items-center gap-2 py-1 text-sm text-[var(--text-secondary)]" role="status">
          <LoaderCircle className="h-4 w-4 animate-spin text-violet-500" /> Mari đang tìm hướng khám phá tiếp...
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
      {showDebug && debug ? (
        <details className="mt-2 rounded-lg border border-violet-200 bg-violet-50/70 px-3 py-2 text-[11px] text-violet-900">
          <summary className="cursor-pointer font-medium">Socratic diagnostics · {debug.status}</summary>
          <div className="mt-2 grid gap-1 sm:grid-cols-2">
            <span>Passages inspected: {debug.passages_scanned ?? debug.passages_inspected?.length ?? 0}</span>
            <span>Opportunities found: {debug.semantic_opportunities?.detected ?? 0}</span>
            <span>Candidates generated: {debug.candidate_generation.generated ?? 0}</span>
            <span>Accepted: {debug.final_accepted ?? debug.candidate_generation.final ?? 0}</span>
          </div>
          {Object.keys(debug.rejection_distribution).length ? (
            <p className="mt-1">Rejected: {Object.entries(debug.rejection_distribution).map(([reason, count]) => `${reason}: ${count}`).join(' · ')}</p>
          ) : null}
        </details>
      ) : null}
    </section>
  );
}
