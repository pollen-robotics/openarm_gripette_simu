"""Collect a reach-and-touch dataset in LeRobot v3 format.

Generates randomized reach episodes and writes them directly as a
LeRobot dataset (parquet + video) ready for training.

Features match the Grabette pipeline:
  - observation.state: [11] float32 — [x, y, z, r6d_0..5, proximal_deg, distal_deg]
  - observation.images.cam0: video — Gripette camera (fisheye, 972x1296)
  - action: [11] float32 — same 11D (absolute, next-step target)

Install: uv sync --extra dataset
Usage:
    uv run python examples/collect_reach_dataset.py \
        --repo-id user/simu_reach --episodes 500
    uv run python examples/collect_reach_dataset.py \
        --repo-id user/simu_reach --episodes 500 --push
    uv run python examples/collect_reach_dataset.py \
        --repo-id user/simu_reach --episodes 10 --viewer
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

from openarm_gripette_simu import Simulation, Kinematics
from openarm_gripette_simu.kinematics import GRIPPER_FRAME
from openarm_gripette_simu.rotation import rotation_matrix_to_6d, rotation_6d_to_matrix

logger = logging.getLogger(__name__)

SCENE = Path(__file__).parent.parent / "scenes" / "table_red_cube.xml"

# Start config
START_JOINTS = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0])

# Cube nominal position (matches table_red_cube.xml)
CUBE_NOMINAL_X = 0.45
CUBE_NOMINAL_Y = -0.2

# Randomization
CUBE_X_NOISE = 0.06
CUBE_Y_NOISE = 0.08
CUBE_YAW_NOISE = np.pi
ARM_JOINT_NOISE = 0.08

# Table bounds
TABLE_X_MIN = 0.165
TABLE_X_MAX = 0.735
TABLE_Y_MIN = -0.285
TABLE_Y_MAX = 0.285

# Timing
FPS = 10  # dataset FPS (one recorded frame per N sim steps)
STEPS_REACH = 80
STEPS_HOLD = 20
SIM_SUBSTEPS = 5
SETTLE_STEPS = 200

# Feature names
STATE_NAMES = ["x", "y", "z", "r6d_0", "r6d_1", "r6d_2",
               "r6d_3", "r6d_4", "r6d_5", "proximal", "distal"]


def fk_state(kin, arm_q, gripper_rad):
    """Compute 11D state from camera frame."""
    T = kin.forward(arm_q, "camera")
    pos = T[:3, 3]
    r6d = rotation_matrix_to_6d(T[:3, :3])
    return np.concatenate([pos, r6d, np.degrees(gripper_rad)]).astype(np.float32)


def has_table_collision(sim):
    """Check if any robot geom touches the table."""
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        name1 = sim.model.geom(c.geom1).name
        name2 = sim.model.geom(c.geom2).name
        is_table = "table" in name1 or "leg" in name1 or "table" in name2 or "leg" in name2
        is_cube = "red_cube" in name1 or "red_cube" in name2
        if is_table and not is_cube:
            return True
    return False


def run_episode(sim, kin, rng, viewer=None):
    """Run one reach-and-touch episode.

    Returns (frames, success, table_collision) where frames is a list of
    dicts with 'state', 'action', 'image' keys.
    """
    # --- Randomize cube ---
    cube_jnt_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
    cube_qadr = sim.model.jnt_qposadr[cube_jnt_id]

    cube_x = np.clip(
        CUBE_NOMINAL_X + rng.uniform(-CUBE_X_NOISE, CUBE_X_NOISE),
        TABLE_X_MIN + 0.02, TABLE_X_MAX - 0.02,
    )
    cube_y = np.clip(
        CUBE_NOMINAL_Y + rng.uniform(-CUBE_Y_NOISE, CUBE_Y_NOISE),
        TABLE_Y_MIN + 0.02, TABLE_Y_MAX - 0.02,
    )
    cube_z = 0.415
    yaw = rng.uniform(-CUBE_YAW_NOISE, CUBE_YAW_NOISE)

    sim.data.qpos[cube_qadr:cube_qadr + 3] = [cube_x, cube_y, cube_z]
    sim.data.qpos[cube_qadr + 3:cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    # --- Randomize arm ---
    init_joints = START_JOINTS + rng.uniform(-ARM_JOINT_NOISE, ARM_JOINT_NOISE, size=7)
    sim.reset_arm(init_joints)
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)
    for _ in range(SETTLE_STEPS):
        sim.step()
    if viewer:
        viewer.sync()

    arm_q = sim.get_arm_positions()
    T_grip = kin.forward(arm_q, frame=GRIPPER_FRAME)
    grip_r6d = rotation_matrix_to_6d(T_grip[:3, :3])
    target_pos = np.array([cube_x, cube_y, T_grip[2, 3]])

    cube_pos_start = sim.data.body("red_cube").xpos.copy()

    # --- Run trajectory ---
    frames = []
    table_collision = False
    all_steps = STEPS_REACH + STEPS_HOLD

    T_grip_now = kin.forward(arm_q, frame=GRIPPER_FRAME)
    start_pos = T_grip_now[:3, 3].copy()
    start_r6d = rotation_matrix_to_6d(T_grip_now[:3, :3])

    for step in range(all_steps):
        # Progress: ramp to 1.0 during reach, hold at 1.0
        if step < STEPS_REACH:
            t = (step + 1) / STEPS_REACH
        else:
            t = 1.0

        # Record observation
        grip_actual = np.array([
            sim.data.joint("proximal").qpos[0],
            sim.data.joint("distal").qpos[0],
        ])
        state = fk_state(kin, arm_q, grip_actual)

        # Render camera (RGB numpy → PIL Image for LeRobot)
        img_rgb = sim.render_camera()
        image = Image.fromarray(img_rgb)

        frames.append({"state": state, "image": image})

        # Interpolate and command
        interp_pos = start_pos + t * (target_pos - start_pos)
        interp_r6d = start_r6d + t * (grip_r6d - start_r6d)
        T_ik = np.eye(4)
        T_ik[:3, :3] = rotation_6d_to_matrix(interp_r6d)
        T_ik[:3, 3] = interp_pos
        arm_q = kin.inverse(T_ik, current_joint_positions=arm_q, n_iter=100,
                            frame=GRIPPER_FRAME)
        sim.set_arm_commands(arm_q)
        for _ in range(SIM_SUBSTEPS):
            sim.step()
        if viewer:
            viewer.sync()

        if has_table_collision(sim):
            table_collision = True

        arm_q = sim.get_arm_positions()

    # Build actions: action[t] = state[t+1] (absolute next-step target)
    for i in range(len(frames) - 1):
        frames[i]["action"] = frames[i + 1]["state"].copy()
    frames[-1]["action"] = frames[-1]["state"].copy()

    # Success: cube moved
    cube_pos_end = sim.data.body("red_cube").xpos.copy()
    cube_moved = np.linalg.norm(cube_pos_end[:2] - cube_pos_start[:2]) > 0.003
    success = cube_moved and not table_collision

    return frames, success, table_collision


def main():
    parser = argparse.ArgumentParser(description="Collect reach dataset in LeRobot format")
    parser.add_argument("--repo-id", type=str, required=True, help="LeRobot dataset repo ID")
    parser.add_argument("--episodes", type=int, default=500, help="Target number of successful episodes")
    parser.add_argument("--root", type=str, default=None, help="Local dataset root (default: HF cache)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--push", action="store_true", help="Push dataset to HuggingFace Hub after collection")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Import lerobot (optional dependency)
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError:
        logger.error("lerobot not installed. Run: uv sync --extra dataset")
        return

    rng = np.random.default_rng(args.seed)
    sim = Simulation(scene_xml=SCENE)
    kin = Kinematics()

    viewer = None
    if args.viewer:
        viewer = sim.launch_passive_viewer()

    # Create the LeRobot dataset
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (11,),
            "names": STATE_NAMES,
        },
        "observation.images.cam0": {
            "dtype": "video",
            "shape": (3, 972, 1296),
            "names": ["channels", "height", "width"],
        },
        "action": {
            "dtype": "float32",
            "shape": (11,),
            "names": STATE_NAMES,
        },
    }

    root = Path(args.root) if args.root else None
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        features=features,
        root=root,
        robot_type="openarm_gripette",
        use_videos=True,
    )

    logger.info(f"Collecting {args.episodes} successful episodes -> {ds.root}")
    n_success = 0
    n_collision = 0
    n_attempted = 0

    t_start = time.perf_counter()

    while n_success < args.episodes:
        frames, success, table_collision = run_episode(sim, kin, rng, viewer)
        n_attempted += 1

        if table_collision:
            n_collision += 1
            if n_attempted % 50 == 0:
                logger.info(f"  attempted {n_attempted}: {n_success} ok, {n_collision} collisions")
            continue

        # Write episode to LeRobot dataset
        for frame in frames:
            ds.add_frame({
                "observation.state": frame["state"],
                "observation.images.cam0": frame["image"],
                "action": frame["action"],
                "task": "reach_red_cube",
            })
        ds.save_episode()
        n_success += 1

        if n_success % 10 == 0:
            elapsed = time.perf_counter() - t_start
            eps = n_success / elapsed
            eta = (args.episodes - n_success) / eps if eps > 0 else 0
            logger.info(
                f"  {n_success}/{args.episodes} episodes "
                f"({n_attempted} attempted, {n_collision} collisions, "
                f"{eps:.1f} ep/s, ETA {eta:.0f}s)"
            )

    ds.finalize()

    elapsed = time.perf_counter() - t_start
    logger.info(
        f"Done. {n_success} episodes saved to {ds.root} "
        f"({n_attempted} attempted, {n_collision} collisions, {elapsed:.0f}s)"
    )

    if args.push:
        logger.info(f"Pushing to HuggingFace Hub as {args.repo_id}...")
        ds.push_to_hub()
        logger.info("Push complete.")

    if viewer:
        viewer.close()


if __name__ == "__main__":
    main()
