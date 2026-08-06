import { motion } from 'framer-motion';

interface VoiceWaveformProps {
  amplitude: number;
  active: boolean;
}

export function VoiceWaveform({ amplitude, active }: VoiceWaveformProps) {
  const bars = Array.from({ length: 24 }, (_, index) => index);

  return (
    <div className="flex h-8 items-end gap-[3px] overflow-hidden rounded-full border border-white/10 bg-white/5 px-3 py-2">
      {bars.map((bar) => {
        const height = active ? 6 + ((bar * 11 + amplitude * 100) % 18) : 5 + (bar % 6);
        return (
          <motion.span
            key={bar}
            className="w-1 rounded-full bg-gradient-to-t from-viqa-violet via-viqa-cyan to-viqa-gold"
            animate={{ height }}
            transition={{ duration: 0.15, ease: 'easeOut' }}
            style={{ opacity: active ? 0.95 : 0.5 }}
          />
        );
      })}
    </div>
  );
}