"""Collect synthetic reach-and-touch trajectories for policy training.

The arm reaches forward to touch a red cube on the table. Cube position
and arm start are randomized per episode. Uses the camera frame for state
reporting and the gripper frame for targeting.

Scene: table_red_cube.xml

Output per episode:
  - states.npy: (T, 11) observation states [x, y, z, r6d_0..5, proximal_deg, distal_deg]
  - actions.npy: (T, 11) actions (= state at next timestep)
  - images/0000.jpg .. NNNN.jpg: JPEG camera frames

Usage:
    uv run python examples/collect_grasp_data.py --episodes 100 --output data/reach
    uv run python examples/collect_grasp_data.py --episodes 5 --viewer
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from openarm_gripette_simu import Simulation, Kinematics
from openarm_gripette_simu.kinematics import GRIPPER_FRAME
from openarm_gripette_simu.rotation import rotation_matrix_to_6d, rotation_6d_to_matrix

logger = logging.getLogger(__name__)

SCENE = Path(__file__).parent.parent / "scenes" / "table_red_cube.xml"

# Start config
# START_JOINTS = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0])
START_JOINTS = np.array([0.35, 0.0, 0.0, -1.81, 0.0, 0.0, 0.0])

# Nominal cube position (matches table_red_cube.xml)
CUBE_NOMINAL_X = 0.40
CUBE_NOMINAL_Y = -0.15

# Randomization
CUBE_X_NOISE = 0.06     # ±6cm in X (limited by arm reach)
CUBE_Y_NOISE = 0.2     # ±8cm in Y (limited by table edge)
CUBE_YAW_NOISE = np.pi  # full rotation around Z
ARM_JOINT_NOISE = 0.08  # ±0.08 rad arm start

# Table bounds (from scene XML: table pos=[0.45,0], size=[0.30,0.30])
TABLE_X_MIN = 0.165
TABLE_X_MAX = 0.735
TABLE_Y_MIN = -0.285
TABLE_Y_MAX = 0.285

# Timing — consistent with 50fps (20ms per frame)
STEPS_REACH = 80       # steps to reach the cube
STEPS_HOLD = 20        # steps holding at the cube
SIM_SUBSTEPS = 10      # 50fps → 20ms per frame → 10 steps at dt=0.002
SETTLE_STEPS = 200


def fk_state(kin, arm_q, gripper_rad):
    """Compute 11D state from camera frame."""
    T = kin.forward(arm_q, "camera")
    pos = T[:3, 3]
    r6d = rotation_matrix_to_6d(T[:3, :3])
    return np.concatenate([pos, r6d, np.degrees(gripper_rad)])


def has_table_collision(sim):
    """Check if any robot geom is in contact with the table."""
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        name1 = sim.model.geom(c.geom1).name
        name2 = sim.model.geom(c.geom2).name
        is_table = "table" in name1 or "leg" in name1 or "table" in name2 or "leg" in name2
        is_cube = "red_cube" in name1 or "red_cube" in name2
        if is_table and not is_cube:
            return True
    return False


def run_phase(sim, kin, arm_q, target_pos, target_r6d, n_steps, states, images, viewer):
    """Interpolate the gripper frame to a target, recording at each step.

    Returns (arm_q, table_collision).
    """
    T_grip_now = kin.forward(arm_q, frame=GRIPPER_FRAME)
    start_pos = T_grip_now[:3, 3].copy()
    start_r6d = rotation_matrix_to_6d(T_grip_now[:3, :3])
    collision = False

    for i in range(n_steps):
        t = (i + 1) / n_steps

        # Record observation before action
        grip_actual = np.array([
            sim.data.joint("proximal").qpos[0],
            sim.data.joint("distal").qpos[0],
        ])
        states.append(fk_state(kin, arm_q, grip_actual))

        img_rgb = sim.render_camera()
        _, jpeg = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
        images.append(jpeg.tobytes())

        # Interpolate gripper target
        interp_pos = start_pos + t * (target_pos - start_pos)
        interp_r6d = start_r6d + t * (target_r6d - start_r6d)

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
            collision = True

        arm_q = sim.get_arm_positions()

    return arm_q, collision


def run_episode(sim, kin, rng, viewer=None):
    """Run one reach-and-touch episode."""
    # --- Randomize cube position ---
    cube_jnt_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
    cube_qadr = sim.model.jnt_qposadr[cube_jnt_id]

    cube_x = CUBE_NOMINAL_X + rng.uniform(-CUBE_X_NOISE, CUBE_X_NOISE)
    cube_y = CUBE_NOMINAL_Y + rng.uniform(-CUBE_Y_NOISE, CUBE_Y_NOISE)
    # Clamp to table bounds (with margin for cube half-size 0.015)
    cube_x = np.clip(cube_x, TABLE_X_MIN + 0.02, TABLE_X_MAX - 0.02)
    cube_y = np.clip(cube_y, TABLE_Y_MIN + 0.02, TABLE_Y_MAX - 0.02)
    cube_z = 0.415  # on the table

    # Random yaw rotation (quaternion for rotation around Z)
    yaw = rng.uniform(-CUBE_YAW_NOISE, CUBE_YAW_NOISE)
    qw, qz = np.cos(yaw / 2), np.sin(yaw / 2)

    sim.data.qpos[cube_qadr:cube_qadr + 3] = [cube_x, cube_y, cube_z]
    sim.data.qpos[cube_qadr + 3:cube_qadr + 7] = [qw, 0, 0, qz]

    # --- Randomize arm start ---
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

    # Target: gripper at the cube position (same height as gripper)
    target_pos = np.array([cube_x, cube_y, T_grip[2, 3]])

    # Record initial cube position
    cube_pos_start = sim.data.body("red_cube").xpos.copy()

    states = []
    images = []
    table_collision = False

    # Phase 1: reach to cube
    arm_q, col = run_phase(sim, kin, arm_q, target_pos, grip_r6d, STEPS_REACH,
                           states, images, viewer)
    table_collision |= col

    # Phase 2: hold at cube
    arm_q, col = run_phase(sim, kin, arm_q, target_pos, grip_r6d, STEPS_HOLD,
                           states, images, viewer)
    table_collision |= col

    # Build actions: action[t] = state[t+1]
    states = np.array(states)
    actions = np.empty_like(states)
    actions[:-1] = states[1:]
    actions[-1] = states[-1]

    # Success: cube moved (was touched) and no table collision
    cube_pos_end = sim.data.body("red_cube").xpos.copy()
    cube_moved = np.linalg.norm(cube_pos_end[:2] - cube_pos_start[:2]) > 0.003
    success = cube_moved and not table_collision

    return states, actions, images, success, table_collision


def main():
    parser = argparse.ArgumentParser(description="Collect synthetic reach data")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default="data/reach")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = Simulation(scene_xml=SCENE)
    kin = Kinematics()

    viewer = None
    if args.viewer:
        viewer = sim.launch_passive_viewer()

    logger.info(f"Collecting {args.episodes} episodes -> {output_dir}")
    n_success = 0
    n_collision = 0
    n_saved = 0

    for ep in range(args.episodes):
        t0 = time.perf_counter()
        states, actions, images, success, table_collision = run_episode(sim, kin, rng, viewer)
        dt = time.perf_counter() - t0

        if table_collision:
            n_collision += 1
            logger.info(f"  episode {ep:4d}: {dt:.1f}s, TABLE COLLISION — discarded")
            continue

        n_success += success

        ep_dir = output_dir / f"episode_{n_saved:04d}"
        ep_dir.mkdir(exist_ok=True)
        np.save(ep_dir / "states.npy", states)
        np.save(ep_dir / "actions.npy", actions)

        img_dir = ep_dir / "images"
        img_dir.mkdir(exist_ok=True)
        for i, jpeg in enumerate(images):
            (img_dir / f"{i:04d}.jpg").write_bytes(jpeg)

        meta = {
            "n_steps": len(states),
            "state_dim": int(states.shape[1]),
            "action_dim": int(actions.shape[1]),
            "success": bool(success),
            "state_names": ["x", "y", "z", "r6d_0", "r6d_1", "r6d_2",
                            "r6d_3", "r6d_4", "r6d_5", "proximal", "distal"],
        }
        (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        status = "TOUCH" if success else "MISS"
        logger.info(f"  episode {ep:4d}: {len(states)} steps, {dt:.1f}s, {status}")
        n_saved += 1

    logger.info(
        f"Done. {n_saved} episodes saved ({n_success} touch, "
        f"{n_saved - n_success} miss, {n_collision} discarded for collision)"
    )
    if viewer:
        viewer.close()


if __name__ == "__main__":
    main()
