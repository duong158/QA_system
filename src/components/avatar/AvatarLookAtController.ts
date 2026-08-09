import type { VRM } from '@pixiv/three-vrm';
import { VRMHumanBoneName } from '@pixiv/three-vrm';
import * as THREE from 'three';
import type { AvatarState } from '@/types/avatar';

export const LOOK_AT_CONFIG = {
  headYawDegrees: 22,
  headPitchDegrees: 14,
  eyeYawDegrees: 28,
  eyePitchDegrees: 17,
  eyeDamping: 11,
  headDamping: 6.5,
  neckDamping: 4.8,
} as const;

const HEAD_CAMERA_LIFT = THREE.MathUtils.degToRad(-7);
const NECK_CAMERA_LIFT = THREE.MathUtils.degToRad(-3);

const STATE_POINTER_INFLUENCE: Record<AvatarState, number> = {
  idle: 1,
  listening: 0.9,
  typing: 0.55,
  retrieving: 0.35,
  reading: 0.25,
  thinking: 0.3,
  speaking: 0.85,
  success: 0.75,
  'no-answer': 0.45,
  error: 0.4,
};

const WORKING_STATES = new Set<AvatarState>(['typing', 'retrieving', 'reading', 'thinking']);

export class AvatarLookAtController {
  private headYaw = 0;
  private headPitch = 0;
  private neckYaw = 0;
  private neckPitch = 0;
  private eyeYaw = 0;
  private eyePitch = 0;

  update(vrm: VRM, state: AvatarState, pointer: THREE.Vector2, elapsed: number, delta: number): void {
    const influence = STATE_POINTER_INFLUENCE[state];
    const working = WORKING_STATES.has(state);
    const yawBias = state === 'reading' ? -3 : working ? -1.5 : 0;
    const pitchBias = working ? 3 : 0;
    const idleDrift = state === 'idle' ? Math.sin(elapsed * 0.42) * 1.2 : 0;

    const targetHeadYawDegrees = THREE.MathUtils.clamp(
      pointer.x * LOOK_AT_CONFIG.headYawDegrees * influence + yawBias + idleDrift,
      -LOOK_AT_CONFIG.headYawDegrees,
      LOOK_AT_CONFIG.headYawDegrees,
    );
    const targetHeadPitchDegrees = THREE.MathUtils.clamp(
      pointer.y * LOOK_AT_CONFIG.headPitchDegrees * influence + pitchBias,
      -LOOK_AT_CONFIG.headPitchDegrees,
      LOOK_AT_CONFIG.headPitchDegrees,
    );
    const eyeInfluence = Math.max(0.68, influence);
    const targetEyeYaw = THREE.MathUtils.clamp(
      pointer.x * LOOK_AT_CONFIG.eyeYawDegrees * eyeInfluence + yawBias,
      -LOOK_AT_CONFIG.eyeYawDegrees,
      LOOK_AT_CONFIG.eyeYawDegrees,
    );
    const targetEyePitch = THREE.MathUtils.clamp(
      pointer.y * LOOK_AT_CONFIG.eyePitchDegrees * eyeInfluence + pitchBias,
      -LOOK_AT_CONFIG.eyePitchDegrees,
      LOOK_AT_CONFIG.eyePitchDegrees,
    );

    const targetYaw = THREE.MathUtils.degToRad(targetHeadYawDegrees);
    const targetPitch = THREE.MathUtils.degToRad(targetHeadPitchDegrees);
    this.headYaw = THREE.MathUtils.damp(this.headYaw, targetYaw * 0.76, LOOK_AT_CONFIG.headDamping, delta);
    this.headPitch = THREE.MathUtils.damp(this.headPitch, targetPitch * 0.78, LOOK_AT_CONFIG.headDamping, delta);
    this.neckYaw = THREE.MathUtils.damp(this.neckYaw, targetYaw * 0.24, LOOK_AT_CONFIG.neckDamping, delta);
    this.neckPitch = THREE.MathUtils.damp(this.neckPitch, targetPitch * 0.22, LOOK_AT_CONFIG.neckDamping, delta);
    this.eyeYaw = THREE.MathUtils.damp(this.eyeYaw, targetEyeYaw, LOOK_AT_CONFIG.eyeDamping, delta);
    this.eyePitch = THREE.MathUtils.damp(this.eyePitch, targetEyePitch, LOOK_AT_CONFIG.eyeDamping, delta);

    const head = vrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.Head);
    const neck = vrm.humanoid.getNormalizedBoneNode(VRMHumanBoneName.Neck);
    const listenTilt = state === 'listening' ? THREE.MathUtils.degToRad(-4) : 0;
    const thinkingTilt = working ? THREE.MathUtils.degToRad(-3.5) : 0;
    const noAnswerShake = state === 'no-answer' ? Math.sin(elapsed * 3.2) * THREE.MathUtils.degToRad(1.4) : 0;

    if (head) {
      head.rotation.y = this.headYaw + noAnswerShake;
      head.rotation.x = HEAD_CAMERA_LIFT + this.headPitch + Math.sin(elapsed * 0.7) * 0.006;
      head.rotation.z = THREE.MathUtils.damp(
        head.rotation.z,
        listenTilt + thinkingTilt,
        LOOK_AT_CONFIG.headDamping,
        delta,
      );
    }
    if (neck) {
      neck.rotation.y = this.neckYaw;
      neck.rotation.x = NECK_CAMERA_LIFT + this.neckPitch;
    }

    if (vrm.lookAt) {
      vrm.lookAt.autoUpdate = false;
      vrm.lookAt.yaw = this.eyeYaw;
      vrm.lookAt.pitch = this.eyePitch;
    }
  }

  getHeadYaw(): number {
    return this.headYaw + this.neckYaw;
  }

  getHeadPitch(): number {
    return this.headPitch + this.neckPitch;
  }
}
