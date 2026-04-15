"""ArmServicer — gRPC service for delta Cartesian arm control.

Maintains an internal Cartesian target (position + orientation). Delta commands
accumulate on this target, avoiding drift from physics errors. IK solves for
the accumulated target, and MuJoCo tracks the resulting joint commands.

Delta commands: 9D [dx, dy, dz, dr6d_0..dr6d_5] (position + 6D rotation)
"""

import logging
import time
import threading
import numpy as np

from .kinematics import Kinematics
from .rotation import rotation_matrix_to_6d, rotation_6d_to_matrix
from .proto import arm_pb2, arm_pb2_grpc

logger = logging.getLogger(__name__)


class ArmServicer(arm_pb2_grpc.ArmServiceServicer):

    def __init__(self, sim, kin: Kinematics, lock: threading.Lock, start_time: float):
        self._sim = sim
        self._kin = kin
        self._lock = lock
        self._start_time = start_time

        # Internal Cartesian target — initialized from current FK
        self._sync_target_from_sim()

    def _sync_target_from_sim(self):
        """Initialize the internal target from the current sim state."""
        arm_joints = self._sim.get_arm_positions()
        T = self._kin.forward(arm_joints)
        self._target_pos = T[:3, 3].copy()
        self._target_r6d = rotation_matrix_to_6d(T[:3, :3]).copy()

    def _get_state_from_sim(self):
        """Read actual arm state from the simulation (for GetArmState)."""
        arm_joints = self._sim.get_arm_positions()
        T = self._kin.forward(arm_joints)
        pos = T[:3, 3]
        r6d = rotation_matrix_to_6d(T[:3, :3])
        return pos, r6d, arm_joints

    def SendCartesianDelta(self, request, context):
        try:
            delta_pos = np.array([request.dx, request.dy, request.dz])
            delta_r6d = np.array(request.dr6d)

            if len(delta_r6d) != 6:
                return arm_pb2.ArmCommandResponse(
                    success=False, error=f"dr6d must have 6 values, got {len(delta_r6d)}"
                )

            with self._lock:
                # Accumulate deltas on the internal target (not the sim state)
                self._target_pos = self._target_pos + delta_pos
                self._target_r6d = self._target_r6d + delta_r6d

                # Re-orthogonalize the rotation
                target_rot = rotation_6d_to_matrix(self._target_r6d)

                # Build target 4x4 transform
                T_target = np.eye(4)
                T_target[:3, :3] = target_rot
                T_target[:3, 3] = self._target_pos

                # IK from current sim joints (for good initial guess)
                arm_joints = self._sim.get_arm_positions()
                target_joints = self._kin.inverse(T_target, current_joint_positions=arm_joints)
                self._sim.set_arm_commands(target_joints)

                # Clamp internal target to what the IK actually achieved,
                # so unreachable targets don't accumulate past workspace limits
                T_achieved = self._kin.forward(target_joints)
                self._target_pos = T_achieved[:3, 3].copy()
                self._target_r6d = rotation_matrix_to_6d(T_achieved[:3, :3]).copy()

            return arm_pb2.ArmCommandResponse(success=True)

        except Exception as e:
            logger.exception("Cartesian delta command failed")
            return arm_pb2.ArmCommandResponse(success=False, error=str(e))

    def GetArmState(self, request, context):
        with self._lock:
            pos, r6d, arm_joints = self._get_state_from_sim()

        return arm_pb2.ArmState(
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            r6d=r6d.tolist(),
            joint_positions=arm_joints.tolist(),
        )

    def Reset(self, request, context):
        """Teleport the arm and re-sync the internal Cartesian target."""
        try:
            joints = np.array(request.joint_positions)
            if len(joints) != 7:
                return arm_pb2.ArmCommandResponse(
                    success=False, error=f"Expected 7 joint values, got {len(joints)}"
                )
            with self._lock:
                self._sim.reset_arm(joints)
                self._sync_target_from_sim()
            logger.info(f"Arm reset to {joints.tolist()}")
            return arm_pb2.ArmCommandResponse(success=True)
        except Exception as e:
            logger.exception("Reset failed")
            return arm_pb2.ArmCommandResponse(success=False, error=str(e))

    def Ping(self, request, context):
        uptime = time.monotonic() - self._start_time
        return arm_pb2.ArmPingResponse(status="ok", uptime_seconds=uptime)
