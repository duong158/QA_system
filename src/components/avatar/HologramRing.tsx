import { useFrame } from '@react-three/fiber';
import { useRef } from 'react';
import type { Mesh } from 'three';
import type { AvatarState } from '@/types/avatar';

interface HologramRingProps {
  state: AvatarState;
}

export function HologramRing({ state }: HologramRingProps) {
  const ringRef = useRef<Mesh>(null);
  const ringRef2 = useRef<Mesh>(null);

  useFrame((clock, delta) => {
    const elapsed = clock.clock.getElapsedTime();
    if (ringRef.current) {
      ringRef.current.rotation.z += delta * (state === 'retrieving' ? 1.2 : 0.3);
      ringRef.current.scale.setScalar(1 + Math.sin(elapsed * 1.8) * 0.03);
    }
    if (ringRef2.current) {
      ringRef2.current.rotation.z -= delta * (state === 'thinking' ? 0.9 : 0.22);
    }
  });

  const opacity = state === 'error' ? 0.16 : 0.3;
  return (
    <group position={[0, -1.35, 0]}>
      <mesh ref={ringRef} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[2.1, 0.035, 16, 120]} />
        <meshBasicMaterial color={state === 'error' ? '#FB7185' : '#38BDF8'} transparent opacity={opacity} />
      </mesh>
      <mesh ref={ringRef2} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[2.45, 0.02, 16, 120]} />
        <meshBasicMaterial color="#818CF8" transparent opacity={0.18} />
      </mesh>
    </group>
  );
}
