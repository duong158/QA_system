import { motion } from 'framer-motion';
import { PipelineStep } from './PipelineStep';
import type { PipelineState, PipelineStepModel } from '@/types/pipeline';

interface PipelineFlowProps {
  state: PipelineState;
}

function buildSteps(state: PipelineState): PipelineStepModel[] {
  const stepConfig: PipelineStepModel[] = [
    { key: 'question', label: 'Question', status: 'completed', progress: 100, accent: 'white' },
    { key: 'retriever', label: 'Retriever', status: 'waiting', progress: 0, accent: 'cyan' },
    { key: 'reader', label: 'Reader', status: 'waiting', progress: 0, accent: 'violet' },
    { key: 'answer', label: 'Answer', status: 'waiting', progress: 0, accent: 'gold' },
  ];

  if (state === 'retrieving') {
    stepConfig[1] = { ...stepConfig[1], status: 'running', progress: 48 };
  }
  if (state === 'reading') {
    stepConfig[1] = { ...stepConfig[1], status: 'completed', progress: 100 };
    stepConfig[2] = { ...stepConfig[2], status: 'running', progress: 55 };
  }
  if (state === 'extracting') {
    stepConfig[1] = { ...stepConfig[1], status: 'completed', progress: 100 };
    stepConfig[2] = { ...stepConfig[2], status: 'completed', progress: 100 };
    stepConfig[3] = { ...stepConfig[3], status: 'running', progress: 42 };
  }
  if (state === 'completed' || state === 'speaking') {
    stepConfig[1] = { ...stepConfig[1], status: 'completed', progress: 100 };
    stepConfig[2] = { ...stepConfig[2], status: 'completed', progress: 100 };
    stepConfig[3] = { ...stepConfig[3], status: 'completed', progress: 100 };
  }
  if (state === 'no-answer') {
    stepConfig[1] = { ...stepConfig[1], status: 'completed', progress: 100 };
    stepConfig[2] = { ...stepConfig[2], status: 'completed', progress: 100 };
    stepConfig[3] = { ...stepConfig[3], status: 'error', progress: 100 };
  }
  if (state === 'error') {
    stepConfig[1] = { ...stepConfig[1], status: 'error', progress: 100 };
  }

  return stepConfig;
}

export function PipelineFlow({ state }: PipelineFlowProps) {
  const steps = buildSteps(state);

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="viqa-panel rounded-[28px] p-4"
    >
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-slate-500">Pipeline</p>
          <h2 className="mt-1 font-display text-lg tracking-[0.16em] text-slate-50">QUESTION → RETRIEVER → READER → ANSWER</h2>
        </div>
        <div className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">{state.toUpperCase()}</div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-4">
        {steps.map((step) => (
          <PipelineStep key={step.key} step={step} />
        ))}
      </div>
    </motion.section>
  );
}
