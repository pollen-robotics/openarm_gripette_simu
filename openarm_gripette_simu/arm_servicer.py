"""ArmServicer — gRPC service for delta Cartesian arm control.

Maintains an internal Cartesian target (position + orientation). Delta commands
accumulate on this target, avoiding drift from physics errors. IK solves for
the accumulated target, and MuJoCo tracks the resulting joint commands.

Supports episode reset with cube/arm randomization and success detection.
"""

import logging
import time
import threading
import numpy as np
import mujoco

from .kinematics import Kinematics, GRIPPER_FRAME
from .rotation import rotation_matrix_to_6d, rotation_6d_to_matrix
from .proto import arm_pb2, arm_pb2_grpc

logger = logging.getLogger(__name__)

# Default arm start for randomized reset
START_JOINTS = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0])

# Cube nominal position (matches table_red_cube.xml)
CUBE_NOMINAL_X = 0.40
CUBE_NOMINAL_Y = -0.15
CUBE_Z = 0.415

# Randomization ranges
CUBE_X_NOISE = 0.06
CUBE_Y_NOISE = 0.2
CUBE_YAW_NOISE = np.pi
ARM_JOINT_NOISE = 0.08

# Table bounds (from table_red_cube.xml)
TABLE_X_MIN = 0.165
TABLE_X_MAX = 0.735
TABLE_Y_MIN = -0.285
TABLE_Y_MAX = 0.285

# Success threshold: cube displacement in XY (meters)
CUBE_MOVED_THRESHOLD = 0.005


class ArmServicer(arm_pb2_grpc.ArmServiceServicer):

    def __init__(self, sim, kin: Kinematics, lock: threading.Lock, start_time: float):
        self._sim = sim
        self._kin = kin
        self._lock = lock
        self._start_time = start_time
        self._rng = np.random.default_rng()

        # Cube initial position for success tracking (set on reset)
        self._cube_start_xy = np.array([CUBE_NOMINAL_X, CUBE_NOMINAL_Y])

        # Internal Cartesian target — initialized from current FK
        self._sync_target_from_sim()

    def _sync_target_from_sim(self):
        """Initialize the internal target from the current sim state."""
        arm_joints = self._sim.get_arm_positions()
        T = self._kin.forward(arm_joints)
        self._target_pos = T[:3, 3].copy()
        self._target_r6d = rotation_matrix_to_6d(T[:3, :3]).copy()

    def _randomize_cube(self):
        """Randomize cube position and orientation. Returns (x, y, z)."""
        cube_jnt_id = mujoco.mj_name2id(self._sim.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
        cube_qadr = self._sim.model.jnt_qposadr[cube_jnt_id]

        cube_x = np.clip(
            CUBE_NOMINAL_X + self._rng.uniform(-CUBE_X_NOISE, CUBE_X_NOISE),
            TABLE_X_MIN + 0.02, TABLE_X_MAX - 0.02,
        )
        cube_y = np.clip(
            CUBE_NOMINAL_Y + self._rng.uniform(-CUBE_Y_NOISE, CUBE_Y_NOISE),
            TABLE_Y_MIN + 0.02, TABLE_Y_MAX - 0.02,
        )
        yaw = self._rng.uniform(-CUBE_YAW_NOISE, CUBE_YAW_NOISE)

        self._sim.data.qpos[cube_qadr:cube_qadr + 3] = [cube_x, cube_y, CUBE_Z]
        self._sim.data.qpos[cube_qadr + 3:cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

        # Zero cube velocity
        cube_dof_adr = self._sim.model.jnt_dofadr[cube_jnt_id]
        self._sim.data.qvel[cube_dof_adr:cube_dof_adr + 6] = 0

        self._cube_start_xy = np.array([cube_x, cube_y])
        return cube_x, cube_y, CUBE_Z

    def _get_state_from_sim(self):
        """Read actual arm state from the simulation."""
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
                self._target_pos = self._target_pos + delta_pos
                self._target_r6d = self._target_r6d + delta_r6d

                target_rot = rotation_6d_to_matrix(self._target_r6d)
                T_target = np.eye(4)
                T_target[:3, :3] = target_rot
                T_target[:3, 3] = self._target_pos

                arm_joints = self._sim.get_arm_positions()
                target_joints = self._kin.inverse(T_target, current_joint_positions=arm_joints)
                self._sim.set_arm_commands(target_joints)

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
        """Reset the episode: teleport arm + randomize cube."""
        try:
            with self._lock:
                # Randomize cube
                cx, cy, cz = self._randomize_cube()

                # Arm: use provided joints or randomize
                if len(request.joint_positions) == 7:
                    joints = np.array(request.joint_positions)
                else:
                    joints = START_JOINTS + self._rng.uniform(
                        -ARM_JOINT_NOISE, ARM_JOINT_NOISE, size=7
                    )

                self._sim.reset_arm(joints)
                self._sim.data.qvel[:] = 0
                mujoco.mj_forward(self._sim.model, self._sim.data)
                self._sync_target_from_sim()

            logger.info(f"Reset: arm={joints.round(3).tolist()}, cube=[{cx:.3f}, {cy:.3f}]")
            return arm_pb2.ResetResponse(
                success=True, cube_x=cx, cube_y=cy, cube_z=cz,
            )
        except Exception as e:
            logger.exception("Reset failed")
            return arm_pb2.ResetResponse(success=False, error=str(e))

    def GetSuccessStatus(self, request, context):
        """Check if the cube was touched (moved from its initial position)."""
        with self._lock:
            cube_xy = self._sim.data.body("red_cube").xpos[:2].copy()

        displacement = float(np.linalg.norm(cube_xy - self._cube_start_xy))
        goal_reached = displacement > CUBE_MOVED_THRESHOLD

        return arm_pb2.SuccessStatusResponse(
            goal_reached=goal_reached,
            cube_displacement=displacement,
        )

    def Ping(self, request, context):
        uptime = time.monotonic() - self._start_time
        return arm_pb2.ArmPingResponse(status="ok", uptime_seconds=uptime)
