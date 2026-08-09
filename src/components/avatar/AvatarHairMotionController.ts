import type { VRM, VRMSpringBoneJoint } from '@pixiv/three-vrm';
import * as THREE from 'three';

interface HairJointState {
  joint: VRMSpringBoneJoint;
  stiffness: number;
  dragForce: number;
}

const tunedHairJoints = new WeakMap<VRM, HairJointState[]>();

export const HAIR_MOTION_CONFIG = {
  baseStiffnessScale: 0.92,
  baseDragReduction: 0.045,
  speedResponse: 1.8,
  responseDamping: 5.5,
  dynamicStiffnessReduction: 0.1,
  dynamicDragReduction: 0.055,
} as const;

export class AvatarHairMotionController {
  private readonly joints: HairJointState[] = [];
  private previousYaw = 0;
  private previousPitch = 0;
  private response = 0;

  constructor(vrm: VRM) {
    const existing = tunedHairJoints.get(vrm);
    if (existing) {
      this.joints.push(...existing);
      return;
    }
    for (const joint of vrm.springBoneManager?.joints ?? []) {
      if (!joint.bone.name.toLowerCase().includes('hair')) {
        continue;
      }
      this.joints.push({
        joint,
        stiffness: joint.settings.stiffness,
        dragForce: joint.settings.dragForce,
      });
      joint.settings.stiffness *= HAIR_MOTION_CONFIG.baseStiffnessScale;
      joint.settings.dragForce = Math.max(0.18, joint.settings.dragForce - HAIR_MOTION_CONFIG.baseDragReduction);
    }
    tunedHairJoints.set(vrm, this.joints);
    vrm.springBoneManager?.setInitState();
  }

  update(yaw: number, pitch: number, delta: number): void {
    const safeDelta = Math.max(delta, 1 / 120);
    const angularSpeed = Math.hypot(yaw - this.previousYaw, pitch - this.previousPitch) / safeDelta;
    const targetResponse = THREE.MathUtils.clamp(
      angularSpeed / HAIR_MOTION_CONFIG.speedResponse,
      0,
      1,
    );
    this.response = THREE.MathUtils.damp(
      this.response,
      targetResponse,
      HAIR_MOTION_CONFIG.responseDamping,
      delta,
    );
    this.previousYaw = yaw;
    this.previousPitch = pitch;

    for (const entry of this.joints) {
      const baseStiffness = entry.stiffness * HAIR_MOTION_CONFIG.baseStiffnessScale;
      const baseDrag = Math.max(0.18, entry.dragForce - HAIR_MOTION_CONFIG.baseDragReduction);
      entry.joint.settings.stiffness = baseStiffness * (
        1 - HAIR_MOTION_CONFIG.dynamicStiffnessReduction * this.response
      );
      entry.joint.settings.dragForce = Math.max(
        0.16,
        baseDrag - HAIR_MOTION_CONFIG.dynamicDragReduction * this.response,
      );
    }
  }
}
