import { motion } from 'framer-motion';
import { useAppStore } from '@/store/appStore';

export function BackgroundEffects() {
  const lowPerformanceMode = useAppStore((state) => state.settings.display.lowPerformanceMode);
  const particlesEnabled = useAppStore((state) => state.settings.display.particlesEnabled);

  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(88,230,255,0.12),transparent_33%),radial-gradient(circle_at_bottom_right,rgba(159,122,234,0.15),transparent_28%)]" />
      <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(148,163,184,0.04)_1px,transparent_1px),linear-gradient(to_bottom,rgba(148,163,184,0.04)_1px,transparent_1px)] bg-[size:84px_84px] opacity-25" />
      {!lowPerformanceMode && particlesEnabled ? (
        <>
          <motion.div
            className="absolute left-[8%] top-[14%] h-48 w-48 rounded-full bg-viqa-cyan/10 blur-3xl"
            animate={{ y: [0, -18, 0], x: [0, 12, 0] }}
            transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut' }}
          />
          <motion.div
            className="absolute right-[10%] top-[22%] h-72 w-72 rounded-full bg-viqa-violet/10 blur-3xl"
            animate={{ y: [0, 20, 0], x: [0, -14, 0] }}
            transition={{ duration: 14, repeat: Infinity, ease: 'easeInOut' }}
          />
          <motion.div
            className="absolute bottom-[10%] left-[32%] h-56 w-56 rounded-full bg-viqa-gold/8 blur-3xl"
            animate={{ scale: [1, 1.08, 1] }}
            transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }}
          />
        </>
      ) : null}
      <div className="absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-viqa-bg via-viqa-bg/60 to-transparent" />
    </div>
  );
}