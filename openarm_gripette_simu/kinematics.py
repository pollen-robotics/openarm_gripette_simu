"""Placo-based kinematics for the OpenArm right arm.

Provides forward and inverse kinematics targeting the 'camera' frame.
Only computes desired joint positions — no physics, no simulation.
"""

from pathlib import Path
import numpy as np
import placo
from openarm_gripette_model import OPENARM_RIGHT_DIR

# The 7 arm joints in MuJoCo ordering (excludes gripper and mimic joints)
ARM_JOINT_NAMES = [
    "r_arm_pitch",
    "r_arm_roll",
    "r_arm_yaw",
    "r_elbow",
    "r_wrist_yaw",
    "r_wrist_roll",
    "r_wrist_pitch",
]

# Target frame for IK
CAMERA_FRAME = "camera"


class Kinematics:
    """Placo wrapper for FK/IK on the OpenArm right arm."""

    def __init__(self, model_dir: str | Path | None = None):
        model_dir = Path(model_dir) if model_dir else OPENARM_RIGHT_DIR
        self.robot = placo.RobotWrapper(str(model_dir))

        # Set up the IK solver
        self.solver = self.robot.make_solver()
        self.solver.mask_fbase(True)  # base is fixed
        self.solver.enable_joint_limits(True)
        self.solver.dt = 0.01

        # Mask non-arm joints so IK only moves the 7 arm DOFs
        self.solver.mask_dof("proximal")
        self.solver.mask_dof("distal")
        self.solver.mask_dof("r_wrist_roll_mimic")

        # Regularization for solver stability
        self.solver.add_regularization_task(1e-4)

        # Frame task for the camera (target is set later)
        self.robot.update_kinematics()
        T_init = self.robot.get_T_world_frame(CAMERA_FRAME)
        self._frame_task = self.solver.add_frame_task(CAMERA_FRAME, T_init)
        self._frame_task.configure(CAMERA_FRAME, "soft", 1.0)

    def forward(self, joint_positions: np.ndarray) -> np.ndarray:
        """Compute the camera frame pose from arm joint positions.

        Args:
            joint_positions: 7-element array of arm joint angles (rad).

        Returns:
            4x4 homogeneous transform (world -> camera).
        """
        for i, name in enumerate(ARM_JOINT_NAMES):
            self.robot.set_joint(name, joint_positions[i])
        self.robot.update_kinematics()
        return self.robot.get_T_world_frame(CAMERA_FRAME).copy()

    def inverse(
        self,
        target_pose: np.ndarray,
        current_joint_positions: np.ndarray | None = None,
        n_iter: int = 500,
    ) -> np.ndarray:
        """Solve IK for a target camera pose.

        Args:
            target_pose: 4x4 homogeneous transform (world -> camera).
            current_joint_positions: optional 7-element starting config.
                If None, uses the robot's current state.
            n_iter: number of solver iterations.

        Returns:
            7-element array of arm joint angles (rad).
        """
        # Seed the solver with current joint positions
        if current_joint_positions is not None:
            for i, name in enumerate(ARM_JOINT_NAMES):
                self.robot.set_joint(name, current_joint_positions[i])
            self.robot.update_kinematics()

        # Update the frame task target
        self._frame_task.T_world_frame = target_pose

        # Iterate the solver
        for _ in range(n_iter):
            self.solver.solve(True)
            self.robot.update_kinematics()

        return self.get_arm_joints()

    def get_arm_joints(self) -> np.ndarray:
        """Read the current arm joint positions from the Placo model."""
        return np.array([self.robot.get_joint(name) for name in ARM_JOINT_NAMES])
