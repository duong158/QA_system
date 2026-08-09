import { Canvas, useThree } from '@react-three/fiber';
import { Suspense, useEffect } from 'react';
import { motion } from 'framer-motion';
import type { AvatarState } from '@/types/avatar';
import { useAudioAnalyzer } from '@/hooks/useAudioAnalyzer';
import { AvatarEnvironment } from './AvatarEnvironment';
import { AnimeAvatar } from './AnimeAvatar';
import { HologramRing } from './HologramRing';
import { ParticleField } from './ParticleField';
import { AudioVisualizer } from './AudioVisualizer';

interface AvatarSceneProps {
  state: AvatarState;
}

const avatarCameraPosition: [number, number, number] = [0, 0.64, 1.95];
const avatarCameraTarget: [number, number, number] = [0, 0.62, 0];

function AvatarCameraRig() {
  const { camera, size } = useThree();

  useEffect(() => {
    const compact = size.width < 640;
    camera.position.set(0, compact ? 0.62 : avatarCameraPosition[1], compact ? 2.2 : avatarCameraPosition[2]);
    if ('fov' in camera) {
      camera.fov = compact ? 34 : 29;
    }
    camera.lookAt(...avatarCameraTarget);
    camera.updateProjectionMatrix();
  }, [camera, size.width]);

  return null;
}

function statusLabel(state: AvatarState) {
  switch (state) {
    case 'listening':
      return 'Listening';
    case 'typing':
      return 'Formulating question';
    case 'retrieving':
      return 'Searching knowledge base';
    case 'reading':
      return 'Analyzing passages';
    case 'thinking':
      return 'Extracting answer';
    case 'speaking':
      return 'Speaking';
    case 'no-answer':
      return 'No answer';
    case 'error':
      return 'System error';
    default:
      return 'Ready';
  }
}

export function AvatarScene({ state }: AvatarSceneProps) {
  const amplitude = useAudioAnalyzer(state === 'listening' || state === 'speaking' || state === 'thinking' || state === 'retrieving');

  return (
    <motion.section
      initial={{ opacity: 0, x: -18 }}
      animate={{ opacity: 1, x: 0 }}
      className="viqa-panel relative min-h-[320px] overflow-hidden rounded-2xl sm:min-h-[500px] xl:min-h-[620px]"
    >
      <div className="absolute inset-x-0 top-0 z-10 flex items-center justify-between px-5 py-4 text-xs text-slate-400">
        <span className="rounded-full border border-slate-400/15 bg-slate-900/35 px-2.5 py-1 capitalize backdrop-blur-md">{state}</span>
        <span className="font-medium text-slate-300">Mari assistant</span>
      </div>

      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_45%,rgba(56,189,248,0.09),transparent_50%)]" />

      <Canvas dpr={[1, 1.5]} camera={{ position: avatarCameraPosition, fov: 29 }} gl={{ antialias: true, alpha: true }} className="absolute inset-0">
        <Suspense fallback={null}>
          <AvatarCameraRig />
          <AvatarEnvironment />
          <ParticleField state={state} />
          <AnimeAvatar state={state} audioLevel={amplitude} />
          <HologramRing state={state} />
        </Suspense>
      </Canvas>

      <AudioVisualizer amplitude={amplitude} />

      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-[#0F172A] via-[#0F172A]/55 to-transparent px-6 pb-5 pt-16 text-center">
        <p className="font-display text-sm font-medium text-slate-100">{statusLabel(state)}</p>
      </div>
    </motion.section>
  );
}
