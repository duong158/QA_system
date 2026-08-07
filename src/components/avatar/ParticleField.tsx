import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { Points } from 'three';
import type { AvatarState } from '@/types/avatar';

interface ParticleFieldProps {
  state: AvatarState;
}

export function ParticleField({ state }: ParticleFieldProps) {
  const pointsRef = useRef<Points>(null);
  const count = 140;

  const positions = useMemo(() => {
    const array = new Float32Array(count * 3);
    for (let index = 0; index < count; index += 1) {
      const radius = 4 + Math.random() * 5;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.random() * Math.PI;
      array[index * 3] = Math.sin(phi) * Math.cos(theta) * radius;
      array[index * 3 + 1] = (Math.cos(phi) - 0.08) * radius * 0.58;
      array[index * 3 + 2] = Math.sin(phi) * Math.sin(theta) * radius;
    }
    return array;
  }, []);

  useFrame((clock) => {
    const elapsed = clock.clock.getElapsedTime();
    if (pointsRef.current) {
      pointsRef.current.rotation.y = elapsed * 0.03;
      pointsRef.current.rotation.x = Math.sin(elapsed * 0.1) * 0.06;
      pointsRef.current.scale.setScalar(state === 'retrieving' ? 1.08 : 1);
    }
  });

  return (
    <points ref={pointsRef} position={[0, 0.05, -0.35]}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={count} array={positions} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial size={0.06} color={state === 'error' ? '#FB7185' : state === 'speaking' ? '#FFD76A' : '#7dd3fc'} transparent opacity={0.8} sizeAttenuation />
    </points>
  );
}
