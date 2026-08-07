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

const avatarCameraPosition: [number, number, number] = [0, 0.52, 2.28];
const avatarCameraTarget: [number, number, number] = [0, 0.5, 0];

function AvatarCameraRig() {
  const { camera } = useThree();

  useEffect(() => {
    camera.position.set(...avatarCameraPosition);
    camera.lookAt(...avatarCameraTarget);
    camera.updateProjectionMatrix();
  }, [camera]);

  return null;
}

function statusLabel(state: AvatarState) {
  switch (state) {
    case 'listening':
      return 'LISTENING';
    case 'retrieving':
      return 'SEARCHING KNOWLEDGE BASE';
    case 'reading':
      return 'ANALYZING PASSAGES';
    case 'thinking':
      return 'EXTRACTING ANSWER';
    case 'speaking':
      return 'SPEAKING';
    case 'no-answer':
      return 'NO ANSWER';
    case 'error':
      return 'SYSTEM ERROR';
    default:
      return 'READY';
  }
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
        <span>MARI 3D ASSISTANT CORE</span>
      </div>

      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(88,230,255,0.08),transparent_40%)]" />

      <Canvas dpr={[1, 1.5]} camera={{ position: avatarCameraPosition, fov: 30 }} gl={{ antialias: true, alpha: true }} className="absolute inset-0">
        <Suspense fallback={null}>
          <AvatarCameraRig />
          <AvatarEnvironment />
          <ParticleField state={state} />
          <AnimeAvatar state={state} audioLevel={amplitude} />
          <HologramRing state={state} />
        </Suspense>
      </Canvas>

      <AudioVisualizer amplitude={amplitude} />

      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-viqa-bg via-viqa-bg/50 to-transparent px-6 pb-6 pt-16 text-center">
        <p className="font-display text-sm tracking-[0.35em] text-slate-200">{statusLabel(state)}</p>
        <p className="mt-2 text-sm text-slate-400">Mari 3D VRoid Model is loaded from public/models/mari.vrm.</p>
      </div>
    </motion.section>
  );
}
