"""Demo: table scene with a red cube.

Moves the arm to look at the table, then slowly approaches the cube.
Shows how to load a custom scene.
"""

import time
from pathlib import Path
import numpy as np
import cv2
from openarm_gripette_simu import Simulation, Kinematics

SCENE = Path(__file__).parent.parent / "scenes" / "table_red_cube.xml"


def main():
    sim = Simulation(scene_xml=SCENE)
    kin = Kinematics()
    viewer = sim.launch_passive_viewer()

    # Start: arm looking down at the table
    # Pitch forward, elbow bent so the camera points at the table surface
    q_start = np.array([0.3, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0])
    T_start = kin.forward(q_start)
    print(f"Start camera position: {T_start[:3, 3]}")

    # Move to start pose
    sim.set_arm_commands(q_start)
    for _ in range(1000):
        sim.step()
        viewer.sync()
        time.sleep(sim.model.opt.timestep)

    # Get the cube position from the simulation
    cube_pos = sim.data.body("red_cube").xpos.copy()
    print(f"Red cube position: {cube_pos}")

    # Define a target pose: camera above the cube, looking down
    T_target = T_start.copy()
    # Move camera towards the cube (adjust x and z to be above it)
    T_target[0, 3] = cube_pos[0]       # align x with cube
    T_target[2, 3] = cube_pos[2] + 0.15  # 15cm above cube

    target_joints = kin.inverse(T_target, current_joint_positions=q_start)
    T_check = kin.forward(target_joints)
    err = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3])
    print(f"IK target above cube — error: {err * 1000:.1f}mm")

    # Interpolate from start to target
    n_steps = 500
    print("\nMoving above the cube... (press 'q' in camera window to quit)")
    for i in range(n_steps):
        t = i / n_steps
        q_cmd = q_start + t * (target_joints - q_start)
        sim.set_arm_commands(q_cmd)
        sim.step()
        viewer.sync()

        if i % 10 == 0:
            img = sim.render_camera()
            cv2.imshow("Gripette camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return

        time.sleep(sim.model.opt.timestep)

    # Hold position and keep showing camera
    print("Above the cube. Close viewer to exit.")
    while viewer.is_running():
        sim.step()
        viewer.sync()
        img = sim.render_camera()
        cv2.imshow("Gripette camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        time.sleep(sim.model.opt.timestep)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
