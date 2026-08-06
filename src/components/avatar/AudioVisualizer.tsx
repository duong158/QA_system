import { useMemo } from 'react';

interface AudioVisualizerProps {
  amplitude: number;
}

export function AudioVisualizer({ amplitude }: AudioVisualizerProps) {
  const bars = useMemo(() => Array.from({ length: 16 }, (_, index) => index), []);

  return (
    <div className="absolute bottom-5 left-1/2 flex -translate-x-1/2 items-end gap-1 rounded-full border border-white/10 bg-black/20 px-4 py-3 backdrop-blur-xl">
      {bars.map((bar) => {
        const height = 10 + Math.abs(Math.sin(bar * 0.6 + amplitude * 2.4)) * (24 + amplitude * 22);
        return <span key={bar} className="w-1.5 rounded-full bg-gradient-to-t from-viqa-violet via-viqa-cyan to-viqa-gold" style={{ height }} />;
      })}
    </div>
  );
}