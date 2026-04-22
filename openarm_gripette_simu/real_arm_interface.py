"""ROS2 bridge for real OpenArm hardware.

Subscribes to /joint_states and publishes to
/forward_position_controller/commands. Exposes get_arm_positions() and
set_arm_commands() so ArmServicer works unchanged against real hardware.

Requires rclpy — source your ROS2 installation before running:
    source /opt/ros/humble/setup.bash
"""

import threading
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
except ImportError as e:
    raise ImportError(
        "rclpy is required for real hardware mode. "
        "Source your ROS2 installation: source /opt/ros/humble/setup.bash"
    ) from e

# Must match the order expected by forward_position_controller and Kinematics
JOINT_NAMES = [
    "openarm_joint1",
    "openarm_joint2",
    "openarm_joint3",
    "openarm_joint4",
    "openarm_joint5",
    "openarm_joint6",
    "openarm_joint7",
]


class RealArmInterface(Node):
    """ROS2 node that wraps /joint_states + /forward_position_controller/commands.

    Drop-in replacement for the MuJoCo Simulation object from ArmServicer's
    perspective — implements get_arm_positions(), set_arm_commands(), reset_arm().
    """

    def __init__(self):
        super().__init__("arm_grpc_bridge")
        self._positions = np.zeros(7)
        self._received = False
        self._lock = threading.Lock()

        self.create_subscription(JointState, "/joint_states", self._cb, 10)
        self._pub = self.create_publisher(
            Float64MultiArray, "/forward_position_controller/commands", 10
        )

    def _cb(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        with self._lock:
            for i, name in enumerate(JOINT_NAMES):
                if name in name_to_pos:
                    self._positions[i] = name_to_pos[name]
            self._received = True

    def wait_for_first_state(self, timeout: float = 10.0) -> bool:
        """Block until the first /joint_states message arrives."""
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._received:
                    return True
            time.sleep(0.05)
        return False

    def get_arm_positions(self) -> np.ndarray:
        # Negate: hardware positive rotation = local -Z (ros2_control URDF uses axis 0 0 -1),
        # but Placo model uses axis 0 0 1, so the sign conventions are opposite.
        with self._lock:
            return -self._positions.copy()

    def set_arm_commands(self, joints: np.ndarray):
        # Negate back: IK returns joints in Placo convention (axis 0 0 1).
        msg = Float64MultiArray()
        msg.data = (-joints).tolist()
        self._pub.publish(msg)

    def reset_arm(self, joints: np.ndarray):
        # No-op — can't teleport a physical arm. ArmServicer.Reset will
        # re-sync its internal Cartesian target from the actual joint positions.
        pass
