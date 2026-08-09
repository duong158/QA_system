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
        'flex min-w-0 items-center gap-2 rounded-lg border px-2.5 py-2 transition',
        step.status === 'running' && 'border-viqa-cyan/30 bg-viqa-cyan/10',
        step.status === 'completed' && 'border-emerald-400/25 bg-emerald-400/10',
        step.status === 'error' && 'border-rose-400/25 bg-rose-400/10',
        step.status === 'waiting' && 'border-slate-400/10 bg-slate-700/25',
      )}
    >
        <div
          className={cn(
            'flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border',
            step.accent === 'cyan' && 'border-viqa-cyan/30 text-viqa-cyan',
            step.accent === 'violet' && 'border-viqa-violet/30 text-viqa-violet',
            step.accent === 'gold' && 'border-viqa-gold/30 text-viqa-gold',
            step.accent === 'white' && 'border-white/10 text-slate-200',
          )}
        >
          <Icon className={cn('h-4 w-4', step.status === 'running' && 'animate-spin')} />
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-xs font-medium text-slate-200">{step.label}</p>
          <p className="mt-0.5 text-[11px] capitalize text-slate-500">{step.status}</p>
        </div>
    </div>
  );
}
