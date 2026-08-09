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
      className="viqa-panel p-3.5"
    >
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="font-display text-sm font-semibold text-slate-100">Pipeline</h2>
        <div className="rounded-full border border-slate-400/15 bg-slate-700/45 px-2.5 py-1 text-xs capitalize text-slate-300">{state}</div>
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {steps.map((step) => (
          <PipelineStep key={step.key} step={step} />
        ))}
      </div>
    </motion.section>
  );
}
