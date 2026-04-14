"""Basic joint-space control: move each joint one by one.

Demonstrates direct joint commanding without IK.
"""

import time
import numpy as np
import cv2
from openarm_gripette_simu import Simulation
from openarm_gripette_simu.simulation import ARM_ACTUATOR_NAMES


def main():
    sim = Simulation()
    viewer = sim.launch_passive_viewer()

    q_cmd = np.zeros(7)
    joint_targets = [
        ("r_arm_pitch", -0.8),
        ("r_arm_roll", -0.5),
        ("r_arm_yaw", 0.5),
        ("r_elbow", -1.2),
        ("r_wrist_yaw", 0.5),
        ("r_wrist_roll", 0.3),
        ("r_wrist_pitch", -0.5),
    ]

    print("Moving each joint in sequence...")
    for joint_name, target in joint_targets:
        idx = ARM_ACTUATOR_NAMES.index(joint_name)
        q_cmd[idx] = target
        sim.set_arm_commands(q_cmd)
        print(f"  {joint_name} -> {target:.1f} rad")

        # Let it settle
        for step in range(500):
            sim.step()
            viewer.sync()
            if step % 10 == 0:
                img = sim.render_camera()
                cv2.imshow("Gripette camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            time.sleep(sim.model.opt.timestep)

    print("\nAll joints moved. Close the viewer to exit.")
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
