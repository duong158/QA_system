import { useEffect, useState } from 'react';

export function useAudioAnalyzer(active: boolean) {
  const [level, setLevel] = useState(0);

  useEffect(() => {
    if (!active) {
      setLevel(0.08);
      return;
    }

    let frame = 0;
    let animationFrame = 0;

    const tick = () => {
      frame += 1;
      const pulse = Math.sin(frame / 7) * 0.28 + Math.sin(frame / 3.5) * 0.12;
      const noise = ((frame * 17) % 11) / 100;
      setLevel(Math.max(0.06, Math.min(1, 0.42 + pulse + noise)));
      animationFrame = window.requestAnimationFrame(tick);
    };

    animationFrame = window.requestAnimationFrame(tick);

    return () => {
      window.cancelAnimationFrame(animationFrame);
    };
  }, [active]);

  return level;
}