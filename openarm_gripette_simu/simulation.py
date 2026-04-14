"""MuJoCo simulation wrapper for the OpenArm right arm + Gripette.

Handles physics stepping, joint control, state readback, and camera rendering.
Joint behavior (gains, damping, friction) is tuned in the MuJoCo XML model.
"""

from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
from .camera import FisheyeCamera

# Path to the default scene XML
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "openarm_gripette_model" / "openarm_gripette_model" / "openarm_right"
_DEFAULT_SCENE = _MODEL_DIR / "scene.xml"

# Joint names in actuator order (matches robot.xml <actuator> section)
ACTUATOR_NAMES = [
    "r_arm_pitch",
    "r_arm_roll",
    "r_arm_yaw",
    "r_elbow",
    "r_wrist_yaw",
    "r_wrist_roll",
    "r_wrist_pitch",
    "proximal",
    "distal",
]

# 7 arm joints (subset of actuators, excludes gripper)
ARM_ACTUATOR_NAMES = ACTUATOR_NAMES[:7]

# Gripette camera name (as defined in robot.xml)
GRIPETTE_CAM = "gripette_cam"


class Simulation:
    """MuJoCo simulation of the OpenArm + Gripette."""

    def __init__(self, scene_xml: str | Path | None = None):
        scene_xml = Path(scene_xml) if scene_xml else _DEFAULT_SCENE
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)

        # Build actuator name -> index mapping
        self._actuator_ids = {}
        for i in range(self.model.nu):
            name = self.model.actuator(i).name
            self._actuator_ids[name] = i

        # Fisheye camera model (precomputes remap tables)
        self._fisheye = FisheyeCamera()

        # Camera renderer (created lazily, renders at pinhole resolution)
        self._renderer = None
        self._cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, GRIPETTE_CAM)

    def step(self):
        """Advance the simulation by one timestep."""
        mujoco.mj_step(self.model, self.data)

    def set_joint_commands(self, commands: np.ndarray, joint_names: list[str] | None = None):
        """Set position commands for the actuators.

        Args:
            commands: array of target joint positions (rad).
            joint_names: which actuators to command. Defaults to all 9 actuators.
        """
        if joint_names is None:
            joint_names = ACTUATOR_NAMES
        for name, cmd in zip(joint_names, commands):
            self.data.ctrl[self._actuator_ids[name]] = cmd

    def set_arm_commands(self, commands: np.ndarray):
        """Set position commands for the 7 arm joints only."""
        self.set_joint_commands(commands, ARM_ACTUATOR_NAMES)

    def get_joint_positions(self, joint_names: list[str] | None = None) -> np.ndarray:
        """Read current joint positions from the simulation.

        Args:
            joint_names: which joints to read. Defaults to all 9 actuated joints.
        """
        if joint_names is None:
            joint_names = ACTUATOR_NAMES
        return np.array([
            self.data.joint(name).qpos[0] for name in joint_names
        ])

    def get_arm_positions(self) -> np.ndarray:
        """Read current arm joint positions (7 values)."""
        return self.get_joint_positions(ARM_ACTUATOR_NAMES)

    def render_camera(self) -> np.ndarray:
        """Render an image from the Gripette camera with fisheye distortion.

        Renders a wide-FOV pinhole image from MuJoCo, then applies
        KannalaBrandt8 fisheye distortion to match the real camera.

        Returns:
            RGB image as uint8 array of shape (972, 1296, 3).
        """
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self._fisheye.pinhole_height,
                width=self._fisheye.pinhole_width,
            )
        self._renderer.update_scene(self.data, camera=self._cam_id)
        pinhole_img = self._renderer.render()
        return self._fisheye.distort(pinhole_img)

    def launch_viewer(self):
        """Launch the interactive MuJoCo viewer (blocking)."""
        mujoco.viewer.launch(self.model, self.data)

    def launch_passive_viewer(self):
        """Launch a passive MuJoCo viewer (non-blocking).

        Returns the viewer handle. Call viewer.sync() after each step.
        """
        return mujoco.viewer.launch_passive(self.model, self.data)
