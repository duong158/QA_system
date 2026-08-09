import { motion } from 'framer-motion';

interface ProcessingIndicatorProps {
  active: boolean;
  label: string;
}

export function ProcessingIndicator({ active, label }: ProcessingIndicatorProps) {
  if (!active) {
    return null;
  }

  return (
    <div className="flex items-center gap-3 rounded-full border border-viqa-cyan/20 bg-viqa-cyan/10 px-4 py-2 text-xs text-viqa-cyan">
      <motion.span
        className="h-2.5 w-2.5 rounded-full bg-viqa-cyan"
        animate={{ opacity: [0.35, 1, 0.35], scale: [0.9, 1.15, 0.9] }}
        transition={{ duration: 1.2, repeat: Infinity, ease: 'easeInOut' }}
      />
      {label}
    </div>
  );
}
