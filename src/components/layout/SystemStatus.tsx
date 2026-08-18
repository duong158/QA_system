import { motion } from 'framer-motion';
import { useAppStore } from '@/store/appStore';

export function SystemStatus() {
  const status = useAppStore((state) => state.statusMessage);

  return (
    <div
      data-testid="system-status"
      className="flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1.5 text-xs text-emerald-200"
    >
      <motion.span
        className="h-2 w-2 rounded-full bg-emerald-300"
        animate={{ opacity: [0.4, 1, 0.4] }}
        transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
      />
      {status}
    </div>
  );
}
