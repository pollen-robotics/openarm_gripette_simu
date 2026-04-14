"""Demonstrate gripper open/close while moving the arm.

Moves the arm to a pose, then cycles the gripper open and closed.
"""

import time
import numpy as np
import cv2
from openarm_gripette_simu import Simulation, Kinematics
from openarm_gripette_simu.simulation import ACTUATOR_NAMES


def main():
    sim = Simulation()
    kin = Kinematics()
    viewer = sim.launch_passive_viewer()

    # Move arm to a bent pose
    q_arm = np.array([-0.5, -0.3, 0.0, -1.0, 0.0, 0.0, 0.0])
    sim.set_arm_commands(q_arm)
    print("Moving arm to start pose...")
    for _ in range(1000):
        sim.step()
        viewer.sync()
        time.sleep(sim.model.opt.timestep)

    # Gripper open/close cycle
    # proximal: 0 = open, -pi/2 = closed
    # distal: 0 = open, -2.09 = closed
    proximal_idx = ACTUATOR_NAMES.index("proximal")
    distal_idx = ACTUATOR_NAMES.index("distal")

    print("Cycling gripper... (press 'q' in camera window to quit)")
    n_cycles = 0
    while viewer.is_running():
        # Close
        for t in np.linspace(0, 1, 300):
            sim.data.ctrl[proximal_idx] = -1.57 * t
            sim.data.ctrl[distal_idx] = -2.09 * t
            sim.step()
            viewer.sync()
            if int(t * 300) % 10 == 0:
                img = sim.render_camera()
                cv2.imshow("Gripette camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    return
            time.sleep(sim.model.opt.timestep)

        # Open
        for t in np.linspace(1, 0, 300):
            sim.data.ctrl[proximal_idx] = -1.57 * t
            sim.data.ctrl[distal_idx] = -2.09 * t
            sim.step()
            viewer.sync()
            if int(t * 300) % 10 == 0:
                img = sim.render_camera()
                cv2.imshow("Gripette camera", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    return
            time.sleep(sim.model.opt.timestep)

        n_cycles += 1
        print(f"  cycle {n_cycles}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
