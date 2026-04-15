"""Launch the simulation gRPC server.

Usage:
    uv run python -m openarm_gripette_simu
    uv run python -m openarm_gripette_simu --scene scenes/table_red_cube.xml
    uv run python -m openarm_gripette_simu --headless
"""

import argparse
import logging

from .server import SimulationServer, GRIPPER_PORT, ARM_PORT


def main():
    parser = argparse.ArgumentParser(description="OpenArm Gripette simulation server")
    parser.add_argument("--scene", type=str, default=None, help="Path to scene XML file")
    parser.add_argument("--gripper-port", type=int, default=GRIPPER_PORT)
    parser.add_argument("--arm-port", type=int, default=ARM_PORT)
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    server = SimulationServer(scene_xml=args.scene)
    server.run(
        gripper_port=args.gripper_port,
        arm_port=args.arm_port,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
