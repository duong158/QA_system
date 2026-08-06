import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { Group } from 'three';
import type { AvatarState } from '@/types/avatar';

interface AvatarModelProps {
  state: AvatarState;
  amplitude: number;
}

export function AvatarModel({ state, amplitude }: AvatarModelProps) {
  const groupRef = useRef<Group>(null);
  const headRef = useRef<THREE.Mesh>(null);
  const bodyRef = useRef<THREE.Mesh>(null);

  const accent = useMemo(() => {
    switch (state) {
      case 'retrieving':
      case 'listening':
        return '#58E6FF';
      case 'thinking':
      case 'reading':
        return '#9F7AEA';
      case 'speaking':
      case 'success':
        return '#FFD76A';
      case 'no-answer':
        return '#FBBF24';
      case 'error':
        return '#FB7185';
      default:
        return '#A5F3FC';
    }
  }, [state]);

  useFrame((clock, delta) => {
    const elapsed = clock.clock.getElapsedTime();
    if (!groupRef.current || !headRef.current || !bodyRef.current) {
      return;
    }

    groupRef.current.position.y = Math.sin(elapsed * 1.5) * (state === 'idle' ? 0.06 : 0.04);
    groupRef.current.rotation.y = Math.sin(elapsed * 0.35) * 0.18;
    groupRef.current.rotation.x = state === 'listening' ? -0.08 : Math.sin(elapsed * 0.25) * 0.05;
    headRef.current.rotation.z = state === 'listening' ? -0.16 : Math.sin(elapsed * 0.8) * 0.02;
    bodyRef.current.scale.setScalar(1 + amplitude * 0.08);
    headRef.current.scale.setScalar(1 + amplitude * 0.03);

    if (state === 'retrieving') {
      groupRef.current.rotation.y += delta * 0.2;
    }
    if (state === 'thinking') {
      groupRef.current.rotation.y += delta * 0.08;
    }
    if (state === 'speaking') {
      bodyRef.current.rotation.z = Math.sin(elapsed * 4) * 0.03;
    }
    if (state === 'error') {
      groupRef.current.position.x = Math.sin(elapsed * 18) * 0.008;
    }
  });

  return (
    <group ref={groupRef} position={[0, -0.6, 0]}>
      <mesh ref={bodyRef} position={[0, -1.55, 0]} castShadow>
        <capsuleGeometry args={[0.95, 1.05, 12, 24]} />
        <meshStandardMaterial color="#0d1633" metalness={0.45} roughness={0.22} emissive={accent} emissiveIntensity={0.08} />
      </mesh>

      <mesh ref={headRef} position={[0, 0.15, 0]} castShadow>
        <icosahedronGeometry args={[1.15, 1]} />
        <meshStandardMaterial color="#122044" metalness={0.6} roughness={0.18} emissive={accent} emissiveIntensity={0.12} />
      </mesh>

      <mesh position={[0, 0.12, 1.02]}>
        <sphereGeometry args={[0.12, 24, 24]} />
        <meshBasicMaterial color={accent} />
      </mesh>
      <mesh position={[-0.34, 0.14, 0.98]}>
        <sphereGeometry args={[0.06, 24, 24]} />
        <meshBasicMaterial color="#f8fafc" />
      </mesh>
      <mesh position={[0.34, 0.14, 0.98]}>
        <sphereGeometry args={[0.06, 24, 24]} />
        <meshBasicMaterial color="#f8fafc" />
      </mesh>

      <mesh position={[0, -0.75, 0.2]} rotation={[0, 0, 0]}>
        <torusGeometry args={[0.52, 0.08, 16, 48]} />
        <meshBasicMaterial color={accent} transparent opacity={0.65} />
      </mesh>
    </group>
  );
}