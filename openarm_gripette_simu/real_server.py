"""gRPC ArmService server for real OpenArm + Gripette hardware.

Bridges ROS2 (/joint_states + /forward_position_controller/commands) with the
ArmService gRPC interface. The same clients (e.g. cartesian_square_grpc.py)
work against real hardware without modification.

Prerequisites:
    1. ROS2 Humble sourced:
           source /opt/ros/humble/setup.bash
    2. Bringup running with forward_position_controller:
           ros2 launch openarm_bringup openarm_gripette.launch.py \\
               robot_controller:=forward_position_controller

Usage:
    openarm-gripette-real
    openarm-gripette-real --arm-port 50052
"""

import argparse
import logging
import time
import threading
from concurrent import futures

import grpc

from .kinematics import Kinematics
from .arm_servicer import ArmServicer
from .proto import arm_pb2_grpc

ARM_PORT = 50052

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="OpenArm real hardware gRPC server")
    parser.add_argument("--arm-port", type=int, default=ARM_PORT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    # Delayed import — keeps rclpy out of the package unless this entry point is used.
    # Install via: pip install openarm-gripette-simu[ros]  (rclpy provided by ROS2 Humble)
    try:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
    except ImportError:
        raise SystemExit(
            "rclpy not found. Source your ROS2 installation:\n"
            "  source /opt/ros/humble/setup.bash"
        )

    from .real_arm_interface import RealArmInterface

    rclpy.init()
    node = RealArmInterface()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    logger.info("Waiting for /joint_states ...")
    if not node.wait_for_first_state(timeout=10.0):
        rclpy.shutdown()
        raise SystemExit(
            "Timed out waiting for /joint_states.\n"
            "Is the bringup running with robot_controller:=forward_position_controller?"
        )
    logger.info("Joint states received — starting gRPC server.")

    kin = Kinematics()
    lock = threading.Lock()
    servicer = ArmServicer(node, kin, lock, time.monotonic())

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    arm_pb2_grpc.add_ArmServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"0.0.0.0:{args.arm_port}")
    server.start()
    logger.info(f"ArmService listening on port {args.arm_port}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop(grace=2)
        rclpy.shutdown()
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
