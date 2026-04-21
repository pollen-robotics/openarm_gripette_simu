"""Move the arm end-effector along a square trajectory via gRPC.

Connects to ArmService and traces a 20x20cm square in the YZ plane using
SendCartesianDelta commands. Works unchanged against both the simulator
and real hardware — just set SERVER to the target host.
"""

import time
import grpc
import numpy as np
from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc

SERVER = "localhost:50052"
SIDE = 0.20           # square side length in meters
STEPS_PER_EDGE = 50   # delta commands per edge
STEP_DT = 0.05        # seconds between commands (~20 Hz)


def move_to(arm, current: np.ndarray, target: np.ndarray, steps: int):
    """Move from current to target position by sending equal delta steps."""
    delta = (target - current) / steps
    for _ in range(steps):
        arm.SendCartesianDelta(arm_pb2.CartesianDelta(
            dx=float(delta[0]),
            dy=float(delta[1]),
            dz=float(delta[2]),
            dr6d=[0.0] * 6,  # keep orientation fixed
        ))
        time.sleep(STEP_DT)


def main():
    channel = grpc.insecure_channel(SERVER)
    arm = arm_pb2_grpc.ArmServiceStub(channel)

    # Use the current EE position as the square center
    state = arm.GetArmState(arm_pb2.GetArmStateRequest())
    center = np.array([state.x, state.y, state.z])
    print(f"Square center: x={center[0]:.3f}  y={center[1]:.3f}  z={center[2]:.3f}")

    # 4 corners in the YZ plane (X fixed), counter-clockwise
    half = SIDE / 2
    corners = [
        center + np.array([0,  half,  half]),
        center + np.array([0,  half, -half]),
        center + np.array([0, -half, -half]),
        center + np.array([0, -half,  half]),
    ]
    for i, c in enumerate(corners):
        print(f"  corner {i}: y={c[1]:.3f}  z={c[2]:.3f}")

    # Move to the first corner before starting the loop
    print("\nMoving to start corner...")
    move_to(arm, center, corners[0], STEPS_PER_EDGE)

    # Trace the square continuously until Ctrl+C
    print("Tracing square... (Ctrl+C to stop)")
    try:
        while True:
            for i in range(4):
                move_to(arm, corners[i], corners[(i + 1) % 4], STEPS_PER_EDGE)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
