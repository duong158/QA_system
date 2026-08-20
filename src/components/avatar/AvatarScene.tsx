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
  compact?: boolean;
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
      return 'Đang lắng nghe';
    case 'typing':
      return 'Sẵn sàng nhận câu hỏi';
    case 'retrieving':
      return 'Đang tìm trong tài liệu';
    case 'reading':
      return 'Đang đọc nguồn';
    case 'thinking':
      return 'Mari đang suy nghĩ';
    case 'speaking':
      return 'Đang đọc câu trả lời';
    case 'no-answer':
      return 'Chưa tìm thấy câu trả lời';
    case 'error':
      return 'Có lỗi kết nối';
    default:
      return 'Sẵn sàng';
  }
}

export function AvatarScene({ state, compact = false }: AvatarSceneProps) {
  const amplitude = useAudioAnalyzer(state === 'listening' || state === 'speaking' || state === 'thinking' || state === 'retrieving');

  return (
    <motion.section
      initial={{ opacity: 0, x: -18 }}
      animate={{ opacity: 1, x: 0 }}
      className={`surface-card relative overflow-hidden rounded-2xl bg-gradient-to-b from-indigo-50 via-white to-slate-50 ${
        compact ? 'h-[clamp(420px,56vh,500px)] min-h-[420px]' : 'min-h-[320px] sm:min-h-[500px] xl:min-h-[620px]'
      }`}
    >
      <div className="absolute inset-x-0 top-0 z-10 flex items-center justify-between px-4 py-4 text-xs text-slate-500">
        <span className="rounded-full border border-slate-200 bg-white/80 px-2.5 py-1 capitalize shadow-sm backdrop-blur-md">{state}</span>
        <span className="font-medium text-slate-600">Mari</span>
      </div>

      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_45%,rgba(99,102,241,0.10),transparent_52%)]" />

      <Canvas dpr={[1, 1.5]} camera={{ position: avatarCameraPosition, fov: 29 }} gl={{ antialias: true, alpha: true }} className="absolute inset-0">
        <Suspense fallback={null}>
          <AvatarCameraRig />
          <AvatarEnvironment />
          {!compact ? <ParticleField state={state} /> : null}
          <AnimeAvatar state={state} audioLevel={amplitude} />
          {!compact ? <HologramRing state={state} /> : null}
        </Suspense>
      </Canvas>

      {!compact ? <AudioVisualizer amplitude={amplitude} /> : null}

      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-white via-white/70 to-transparent px-6 pb-5 pt-16 text-center">
        <p className="text-sm font-medium text-slate-700">{statusLabel(state)}</p>
      </div>
    </motion.section>
  );
}
