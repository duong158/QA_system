import type { VRM } from '@pixiv/three-vrm';
import { VRMHumanBoneName } from '@pixiv/three-vrm';
import * as THREE from 'three';
import type { AvatarState } from '@/types/avatar';

interface BonePose {
  bone: THREE.Object3D;
  idle: THREE.Quaternion;
  thinking: THREE.Quaternion;
  target: THREE.Quaternion;
}

export const THINKING_POSE_CONFIG = {
  enterDamping: 4.8,
  exitDamping: 6.2,
  boneDamping: 10,
  stateWeights: {
    typing: 0.48,
    retrieving: 0.92,
    reading: 0.84,
    thinking: 1,
  },
} as const;

const THINKING_ROTATIONS: Partial<Record<VRMHumanBoneName, THREE.Euler>> = {
  [VRMHumanBoneName.RightShoulder]: new THREE.Euler(-0.06, -0.1, -0.14, 'XYZ'),
  [VRMHumanBoneName.RightUpperArm]: new THREE.Euler(-0.38, -0.3, -0.58, 'XYZ'),
  [VRMHumanBoneName.RightLowerArm]: new THREE.Euler(0.12, -0.52, -1.42, 'XYZ'),
  [VRMHumanBoneName.RightHand]: new THREE.Euler(-0.18, 0.22, -0.34, 'XYZ'),
};

export class AvatarPoseController {
  private readonly bones: BonePose[] = [];
  private weight = 0;

  constructor(vrm: VRM) {
    for (const [name, rotation] of Object.entries(THINKING_ROTATIONS)) {
      const bone = vrm.humanoid.getNormalizedBoneNode(name as VRMHumanBoneName);
      if (!bone || !rotation) {
        continue;
      }
      this.bones.push({
        bone,
        idle: bone.quaternion.clone(),
        thinking: new THREE.Quaternion().setFromEuler(rotation),
        target: new THREE.Quaternion(),
      });
    }
  }

  update(state: AvatarState, delta: number): void {
    const weights = THINKING_POSE_CONFIG.stateWeights as Partial<Record<AvatarState, number>>;
    const targetWeight = weights[state] ?? 0;
    const damping = targetWeight > this.weight
      ? THINKING_POSE_CONFIG.enterDamping
      : THINKING_POSE_CONFIG.exitDamping;
    this.weight = THREE.MathUtils.damp(this.weight, targetWeight, damping, delta);
    const blend = 1 - Math.exp(-THINKING_POSE_CONFIG.boneDamping * delta);

    for (const pose of this.bones) {
      pose.target.copy(pose.idle).slerp(pose.thinking, this.weight);
      pose.bone.quaternion.slerp(pose.target, blend);
    }
  }
}
