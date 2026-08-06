import { Canvas } from '@react-three/fiber';
import { Suspense } from 'react';
import { motion } from 'framer-motion';
import type { AvatarState } from '@/types/avatar';
import { useAudioAnalyzer } from '@/hooks/useAudioAnalyzer';
import { AvatarEnvironment } from './AvatarEnvironment';
import { AvatarModel } from './AvatarModel';
import { HologramRing } from './HologramRing';
import { ParticleField } from './ParticleField';
import { AudioVisualizer } from './AudioVisualizer';

interface AvatarSceneProps {
  state: AvatarState;
}

export function AvatarScene({ state }: AvatarSceneProps) {
  const amplitude = useAudioAnalyzer(state === 'listening' || state === 'speaking' || state === 'thinking' || state === 'retrieving');

  return (
    <motion.section
      initial={{ opacity: 0, x: -18 }}
      animate={{ opacity: 1, x: 0 }}
      className="viqa-panel relative min-h-[640px] overflow-hidden rounded-[32px]"
    >
      <div className="absolute inset-x-0 top-0 z-10 flex items-center justify-between px-5 py-4 text-xs uppercase tracking-[0.34em] text-slate-500">
        <span>{state.toUpperCase()}</span>
        <span>3D ASSISTANT CORE</span>
      </div>

      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(88,230,255,0.08),transparent_40%)]" />

      <Canvas shadows camera={{ position: [0, 1.2, 8.2], fov: 40 }} gl={{ antialias: true, alpha: true }} className="absolute inset-0">
        <Suspense fallback={null}>
          <AvatarEnvironment />
          <ParticleField state={state} />
          <AvatarModel state={state} amplitude={amplitude} />
          <HologramRing state={state} />
        </Suspense>
      </Canvas>

      <AudioVisualizer amplitude={amplitude} />

      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-viqa-bg via-viqa-bg/50 to-transparent px-6 pb-6 pt-16 text-center">
        <p className="font-display text-sm tracking-[0.35em] text-slate-200">{state === 'listening' ? 'LISTENING' : state === 'retrieving' ? 'SEARCHING KNOWLEDGE BASE' : state === 'reading' ? 'ANALYZING PASSAGES' : state === 'thinking' ? 'EXTRACTING ANSWER' : state === 'speaking' ? 'SPEAKING' : state === 'no-answer' ? 'NO ANSWER' : state === 'error' ? 'SYSTEM ERROR' : 'READY'}</p>
        <p className="mt-2 text-sm text-slate-400">
          Avatar 3D placeholder, sẵn sàng thay thế bằng file GLB thật khi nhóm có model chính thức.
        </p>
      </div>
    </motion.section>
  );
}