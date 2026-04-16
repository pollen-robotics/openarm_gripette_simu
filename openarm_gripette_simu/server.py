"""Simulation gRPC server.

Physics and camera rendering run in the main thread (with optional viewer).
gRPC servers run in background threads and read cached camera frames.
  - GripperService (port 50051): same API as the real Gripette
  - ArmService (port 50052): delta Cartesian arm control
"""

import logging
import time
from concurrent import futures
from pathlib import Path
import threading

import grpc
import mujoco.viewer

from .simulation import Simulation
from .kinematics import Kinematics
from .gripper_servicer import GripperServicer
from .arm_servicer import ArmServicer
from .proto import gripper_pb2_grpc, arm_pb2_grpc

logger = logging.getLogger(__name__)

GRIPPER_PORT = 50051
ARM_PORT = 50052
VIEWER_FPS = 60
CAMERA_FPS = 10  # camera rendering rate (matches real Gripette stream)


class SimulationServer:
    """MuJoCo simulation with gRPC interfaces for the Gripette and arm."""

    def __init__(self, scene_xml: str | Path | None = None, initial_arm_joints=None):
        self._sim = Simulation(scene_xml)
        self._kin = Kinematics()
        self._lock = threading.Lock()
        self._start_time = time.monotonic()

        # Cached camera frame — rendered in the main thread, read by gRPC
        self._camera_frame = self._sim.render_camera()

        if initial_arm_joints is not None:
            import numpy as np
            self._sim.reset_arm(np.asarray(initial_arm_joints, dtype=float))
            logger.info(f"Arm initialized to {list(initial_arm_joints)}")

    def _physics_loop(self):
        """Step physics at the model's timestep rate (background thread)."""
        dt = self._sim.model.opt.timestep
        t_wall = time.perf_counter()
        step = 0
        while self._running:
            with self._lock:
                self._sim.step()
            step += 1
            t_target = t_wall + step * dt
            t_now = time.perf_counter()
            if t_target > t_now:
                time.sleep(t_target - t_now)

    def _start_grpc(self, gripper_port: int, arm_port: int):
        """Start the gRPC servers in background thread pools."""
        self._gripper_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        gripper_pb2_grpc.add_GripperServiceServicer_to_server(
            GripperServicer(self._sim, self, self._lock, self._start_time),
            self._gripper_server,
        )
        self._gripper_server.add_insecure_port(f"0.0.0.0:{gripper_port}")

        self._arm_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        arm_pb2_grpc.add_ArmServiceServicer_to_server(
            ArmServicer(self._sim, self._kin, self._lock, self._start_time),
            self._arm_server,
        )
        self._arm_server.add_insecure_port(f"0.0.0.0:{arm_port}")

        self._gripper_server.start()
        self._arm_server.start()
        logger.info(f"Gripper gRPC server on port {gripper_port}")
        logger.info(f"Arm gRPC server on port {arm_port}")

    def get_camera_frame(self):
        """Get the latest cached camera frame (thread-safe)."""
        return self._camera_frame

    def start(self, gripper_port: int = GRIPPER_PORT, arm_port: int = ARM_PORT):
        """Start gRPC servers + physics in background (non-blocking, headless)."""
        self._start_grpc(gripper_port, arm_port)
        self._running = True
        self._physics_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self._physics_thread.start()
        logger.info("Physics loop running (background).")

    def stop(self):
        """Stop physics and gRPC servers."""
        self._running = False
        if hasattr(self, '_physics_thread'):
            self._physics_thread.join(timeout=2)
        self._gripper_server.stop(grace=2)
        self._arm_server.stop(grace=2)
        logger.info("Server stopped.")

    def run(
        self,
        gripper_port: int = GRIPPER_PORT,
        arm_port: int = ARM_PORT,
        headless: bool = False,
    ):
        """Run the simulation (blocking). Call from the main thread.

        Physics, camera rendering, and viewer sync all happen in the main thread.
        gRPC servers run in background threads and read cached camera frames.
        """
        self._start_grpc(gripper_port, arm_port)

        dt = self._sim.model.opt.timestep
        viewer_interval = max(1, int(1.0 / (VIEWER_FPS * dt)))
        camera_interval = max(1, int(1.0 / (CAMERA_FPS * dt)))

        viewer = None
        if not headless:
            viewer = mujoco.viewer.launch_passive(self._sim.model, self._sim.data)
            logger.info("Viewer launched.")

        logger.info("Physics loop running. Press Ctrl+C to stop.")
        self._running = True
        t_wall = time.perf_counter()
        step = 0

        try:
            while self._running and (viewer is None or viewer.is_running()):
                with self._lock:
                    self._sim.step()

                    # Render camera in the main thread at CAMERA_FPS
                    if step % camera_interval == 0:
                        self._camera_frame = self._sim.render_camera()

                step += 1

                if viewer is not None and step % viewer_interval == 0:
                    viewer.sync()

                t_target = t_wall + step * dt
                t_now = time.perf_counter()
                if t_target > t_now:
                    time.sleep(t_target - t_now)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self._gripper_server.stop(grace=2)
            self._arm_server.stop(grace=2)
            if viewer is not None:
                viewer.close()
            logger.info("Server stopped.")
