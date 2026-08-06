import { Check, Circle, LoaderCircle, TriangleAlert } from 'lucide-react';
import { cn } from '@/utils/cn';
import type { PipelineStepModel } from '@/types/pipeline';

interface PipelineStepProps {
  step: PipelineStepModel;
}

const iconMap = {
  waiting: Circle,
  running: LoaderCircle,
  completed: Check,
  error: TriangleAlert,
};

export function PipelineStep({ step }: PipelineStepProps) {
  const Icon = iconMap[step.status];

  return (
    <div
      className={cn(
        'flex min-w-0 flex-1 flex-col gap-3 rounded-2xl border px-3 py-3 transition',
        step.status === 'running' && 'border-viqa-cyan/30 bg-viqa-cyan/10',
        step.status === 'completed' && 'border-emerald-400/25 bg-emerald-400/10',
        step.status === 'error' && 'border-rose-400/25 bg-rose-400/10',
        step.status === 'waiting' && 'border-white/10 bg-white/5',
      )}
    >
      <div className="flex min-w-0 items-center gap-3">
        <div
          className={cn(
            'flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border',
            step.accent === 'cyan' && 'border-viqa-cyan/30 text-viqa-cyan',
            step.accent === 'violet' && 'border-viqa-violet/30 text-viqa-violet',
            step.accent === 'gold' && 'border-viqa-gold/30 text-viqa-gold',
            step.accent === 'white' && 'border-white/10 text-slate-200',
          )}
        >
          <Icon className={cn('h-4 w-4', step.status === 'running' && 'animate-spin')} />
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-[11px] uppercase tracking-[0.14em] text-slate-300">{step.label}</p>
          <p className="mt-0.5 text-[11px] capitalize text-slate-500">{step.status}</p>
        </div>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-black/30">
        <div
          className={cn(
            'h-full rounded-full transition-all duration-500',
            step.accent === 'cyan' && 'bg-viqa-cyan',
            step.accent === 'violet' && 'bg-viqa-violet',
            step.accent === 'gold' && 'bg-viqa-gold',
            step.accent === 'white' && 'bg-slate-200',
          )}
          style={{ width: `${Math.min(100, Math.max(8, step.progress))}%` }}
        />
      </div>
    </div>
  );
}
