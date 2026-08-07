import { Html } from '@react-three/drei';
import { useFrame, useThree } from '@react-three/fiber';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { MutableRefObject } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRM, VRMLoaderPlugin, VRMHumanBoneName, VRMUtils } from '@pixiv/three-vrm';
import type { AvatarState } from '@/types/avatar';

interface AnimeAvatarProps {
  state: AvatarState;
  audioLevel: number;
}

type LoadState = 'loading' | 'ready' | 'error';

const modelUrl = import.meta.env.VITE_AVATAR_MODEL_URL || '/models/mari.vrm';
let mariPromise: Promise<VRM> | null = null;

const HEAD_CAMERA_LIFT = THREE.MathUtils.degToRad(-7);
const NECK_CAMERA_LIFT = THREE.MathUtils.degToRad(-3);
const CHEST_UPRIGHT_LIFT = THREE.MathUtils.degToRad(-1);
const IDLE_MAX_YAW = THREE.MathUtils.degToRad(6);
const IDLE_MAX_PITCH = THREE.MathUtils.degToRad(3);
const ACTIVE_MAX_YAW = THREE.MathUtils.degToRad(3.5);
const ACTIVE_MAX_PITCH = THREE.MathUtils.degToRad(2);

function loadMariVrm(): Promise<VRM> {
  if (!mariPromise) {
    mariPromise = new Promise((resolve, reject) => {
      const loader = new GLTFLoader();
      loader.register((parser) => new VRMLoaderPlugin(parser));
      loader.load(
        modelUrl,
        (gltf) => {
          const vrm = gltf.userData.vrm as VRM | undefined;
          if (!vrm) {
            reject(new Error('Loaded file is not a VRM model.'));
            return;
          }

          VRMUtils.rotateVRM0(vrm);
          vrm.scene.traverse((object) => {
            object.frustumCulled = false;
          });
          resolve(vrm);
        },
        undefined,
        reject,
      );
    });
  }

  return mariPromise;
}

function setExpression(vrm: VRM | null, name: string, value: number) {
  if (!vrm?.expressionManager) {
    return;
  }

  try {
    vrm.expressionManager.setValue(name, THREE.MathUtils.clamp(value, 0, 1));
  } catch {
    // Some VRM models do not ship every preset expression.
  }
}

function setBoneRotation(vrm: VRM, boneName: VRMHumanBoneName, rotation: THREE.Euler) {
  const bone = vrm.humanoid.getNormalizedBoneNode(boneName);
  if (!bone) {
    return;
  }

  bone.rotation.copy(rotation);
}

function getPointerLookTarget(state: AvatarState, pointer: THREE.Vector2) {
  const directStates: AvatarState[] = ['speaking', 'success', 'thinking'];
  const workingStates: AvatarState[] = ['reading', 'retrieving'];

  if (workingStates.includes(state)) {
    return {
      yaw: THREE.MathUtils.degToRad(-2.5),
      pitch: THREE.MathUtils.degToRad(-1),
    };
  }

  const yawLimit = directStates.includes(state) ? ACTIVE_MAX_YAW : IDLE_MAX_YAW;
  const pitchLimit = directStates.includes(state) ? ACTIVE_MAX_PITCH : IDLE_MAX_PITCH;
  const influence = directStates.includes(state) ? 0.28 : 0.48;

  return {
    yaw: THREE.MathUtils.clamp(pointer.x * 0.1 * influence, -yawLimit, yawLimit),
    pitch: THREE.MathUtils.clamp(-pointer.y * 0.055 * influence, -pitchLimit, pitchLimit),
  };
}

function applyInitialAvatarPose(vrm: VRM) {
  vrm.scene.rotation.set(0, Math.PI, 0);
  vrm.scene.position.set(0, -1.42, 0);
  vrm.scene.scale.setScalar(1.18);

  setBoneRotation(vrm, VRMHumanBoneName.LeftShoulder, new THREE.Euler(0, 0, 0.08));
  setBoneRotation(vrm, VRMHumanBoneName.RightShoulder, new THREE.Euler(0, 0, -0.08));

  setBoneRotation(vrm, VRMHumanBoneName.LeftUpperArm, new THREE.Euler(0.05, 0.02, 1.18));
  setBoneRotation(vrm, VRMHumanBoneName.RightUpperArm, new THREE.Euler(0.05, -0.02, -1.18));

  setBoneRotation(vrm, VRMHumanBoneName.LeftLowerArm, new THREE.Euler(0, 0.08, 0.28));
  setBoneRotation(vrm, VRMHumanBoneName.RightLowerArm, new THREE.Euler(0, -0.08, -0.28));

  setBoneRotation(vrm, VRMHumanBoneName.LeftHand, new THREE.Euler(0, 0, 0.08));
  setBoneRotation(vrm, VRMHumanBoneName.RightHand, new THREE.Euler(0, 0, -0.08));
}

function stateExpression(state: AvatarState) {
  switch (state) {
    case 'listening':
      return 'relaxed';
    case 'reading':
    case 'retrieving':
    case 'thinking':
      return 'neutral';
    case 'speaking':
    case 'success':
      return 'happy';
    case 'no-answer':
      return 'sad';
    case 'error':
      return 'surprised';
    default:
      return 'neutral';
  }
}

function useBlink(vrmRef: MutableRefObject<VRM | null>, blinkRef: MutableRefObject<number>) {
  useEffect(() => {
    let closedTimer: number | undefined;
    let nextTimer: number | undefined;

    const schedule = () => {
      nextTimer = window.setTimeout(
        () => {
          blinkRef.current = 1;
          closedTimer = window.setTimeout(() => {
            blinkRef.current = 0;
            schedule();
          }, 100 + Math.random() * 80);
        },
        2000 + Math.random() * 3000,
      );
    };

    schedule();

    return () => {
      if (nextTimer) {
        window.clearTimeout(nextTimer);
      }
      if (closedTimer) {
        window.clearTimeout(closedTimer);
      }
      setExpression(vrmRef.current, 'blink', 0);
    };
  }, [blinkRef, vrmRef]);
}

export function AnimeAvatar({ state, audioLevel }: AnimeAvatarProps) {
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [vrm, setVrm] = useState<VRM | null>(null);
  const groupRef = useRef<THREE.Group>(null);
  const vrmRef = useRef<VRM | null>(null);
  const blinkRef = useRef(0);
  const mouthRef = useRef(0);
  const expressionValues = useRef<Record<string, number>>({});
  const { pointer } = useThree();

  const expressionNames = useMemo(
    () => ['neutral', 'happy', 'relaxed', 'sad', 'surprised', 'angry', 'aa', 'A', 'mouthOpen', 'blink'],
    [],
  );

  useBlink(vrmRef, blinkRef);

  useEffect(() => {
    let mounted = true;
    setLoadState('loading');

    loadMariVrm()
      .then((loadedVrm) => {
        if (!mounted) {
          return;
        }
        applyInitialAvatarPose(loadedVrm);
        vrmRef.current = loadedVrm;
        setVrm(loadedVrm);
        setLoadState('ready');
      })
      .catch(() => {
        if (mounted) {
          setLoadState('error');
        }
      });

    return () => {
      mounted = false;
      if (groupRef.current && vrmRef.current) {
        groupRef.current.remove(vrmRef.current.scene);
      }
    };
  }, []);

  useFrame((clock, delta) => {
    const currentVrm = vrmRef.current;
    if (!currentVrm) {
      return;
    }

    const elapsed = clock.clock.getElapsedTime();
    const head = currentVrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.Head);
    const chest = currentVrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.Chest);
    const upperChest = currentVrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.UpperChest);
    const neck = currentVrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.Neck);

    if (chest) {
      const breath = Math.sin(elapsed * 1.5) * 0.003;
      chest.position.y = breath;
      chest.rotation.x = THREE.MathUtils.lerp(chest.rotation.x, CHEST_UPRIGHT_LIFT, 0.04);
      chest.rotation.z = THREE.MathUtils.lerp(chest.rotation.z, state === 'no-answer' ? Math.sin(elapsed * 1.7) * 0.01 : 0, 0.05);
    }

    if (upperChest) {
      upperChest.rotation.x = THREE.MathUtils.lerp(upperChest.rotation.x, CHEST_UPRIGHT_LIFT * 0.7, 0.04);
      upperChest.rotation.y = THREE.MathUtils.lerp(upperChest.rotation.y, 0, 0.04);
    }

    if (neck) {
      const readingYaw = state === 'reading' || state === 'retrieving' ? THREE.MathUtils.degToRad(-1.2) : 0;
      neck.rotation.x = THREE.MathUtils.lerp(neck.rotation.x, NECK_CAMERA_LIFT, 0.06);
      neck.rotation.y = THREE.MathUtils.lerp(neck.rotation.y, readingYaw, 0.06);
    }

    if (head) {
      const pointerLook = getPointerLookTarget(state, pointer);
      const listenTilt = state === 'listening' ? THREE.MathUtils.degToRad(-2) : 0;
      const noAnswerShake = state === 'no-answer' ? Math.sin(elapsed * 3.2) * 0.025 : 0;

      head.rotation.y = THREE.MathUtils.lerp(head.rotation.y, pointerLook.yaw + noAnswerShake, 0.08);
      head.rotation.x = THREE.MathUtils.lerp(head.rotation.x, HEAD_CAMERA_LIFT + pointerLook.pitch + Math.sin(elapsed * 0.7) * 0.006, 0.08);
      head.rotation.z = THREE.MathUtils.lerp(head.rotation.z, listenTilt, 0.07);
    }

    const activeExpression = stateExpression(state);
    expressionNames.forEach((name) => {
      const target = name === activeExpression ? (activeExpression === 'neutral' ? 0.15 : 0.48) : 0;
      const next = THREE.MathUtils.lerp(expressionValues.current[name] ?? 0, target, 0.08);
      expressionValues.current[name] = next;
      if (name !== 'aa' && name !== 'A' && name !== 'mouthOpen' && name !== 'blink') {
        setExpression(currentVrm, name, next);
      }
    });

    const mouthTarget = state === 'speaking' ? Math.min(audioLevel * 1.3, 0.8) : 0;
    mouthRef.current = THREE.MathUtils.lerp(mouthRef.current, mouthTarget, 0.3);
    ['aa', 'A', 'mouthOpen'].forEach((name) => setExpression(currentVrm, name, mouthRef.current));

    const blinkTarget = blinkRef.current;
    const blinkValue = THREE.MathUtils.lerp(currentVrm.expressionManager?.getValue('blink') ?? 0, blinkTarget, 0.45);
    setExpression(currentVrm, 'blink', blinkValue);

    currentVrm.update(delta);
  });

  if (loadState === 'error') {
    return (
      <Html center>
        <div className="w-72 rounded-2xl border border-viqa-error/30 bg-black/70 px-5 py-4 text-center text-sm leading-6 text-slate-100 shadow-glow backdrop-blur-xl">
          <p className="font-semibold text-viqa-error">Không thể tải Mari Avatar.</p>
          <p className="mt-2 text-slate-300">Vui lòng đặt file mari.vrm vào public/models/.</p>
        </div>
      </Html>
    );
  }

  if (loadState === 'loading') {
    return (
      <Html center>
        <div className="rounded-full border border-viqa-cyan/25 bg-black/60 px-5 py-3 text-xs uppercase tracking-[0.26em] text-viqa-cyan backdrop-blur-xl">
          Loading Mari
        </div>
      </Html>
    );
  }

  return <group ref={groupRef}>{vrm ? <primitive object={vrm.scene} /> : null}</group>;
}
