"""GripperServicer — simulated Gripette gRPC service.

Mimics the real Gripette API (gripper.proto) using MuJoCo simulation.
Camera frames are rendered from MuJoCo and JPEG-encoded.
Motor commands control the proximal/distal joints.
"""

import logging
import time
import threading
import cv2
import numpy as np

from .proto import gripper_pb2, gripper_pb2_grpc

logger = logging.getLogger(__name__)

STREAM_HZ = 10
STREAM_INTERVAL = 1.0 / STREAM_HZ


class GripperServicer(gripper_pb2_grpc.GripperServiceServicer):

    def __init__(self, sim, lock: threading.Lock, start_time: float):
        self._sim = sim
        self._lock = lock
        self._start_time = start_time

    def _get_motor_positions(self):
        """Read proximal/distal joint positions (rad)."""
        pos = self._sim.get_joint_positions(["proximal", "distal"])
        return float(pos[0]), float(pos[1])

    def StreamState(self, request, context):
        logger.info("StreamState: client connected")
        sequence = 0
        next_time = time.monotonic()

        while context.is_active():
            with self._lock:
                img = self._sim.render_camera()
                pos1, pos2 = self._get_motor_positions()

            # Encode as JPEG (outside lock)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, jpeg_data = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])

            frame = gripper_pb2.GripperFrame(
                jpeg_data=jpeg_data.tobytes(),
                motor_state=gripper_pb2.MotorState(
                    motor1_position=pos1,
                    motor2_position=pos2,
                ),
                timestamp_ms=(time.monotonic() - self._start_time) * 1000.0,
                sequence=sequence,
            )
            yield frame
            sequence += 1

            next_time += STREAM_INTERVAL
            sleep_dur = next_time - time.monotonic()
            if sleep_dur > 0:
                time.sleep(sleep_dur)

        logger.info("StreamState: client disconnected after %d frames", sequence)

    def SendMotorCommand(self, request, context):
        try:
            with self._lock:
                self._sim.set_joint_commands(
                    np.array([request.motor1_goal, request.motor2_goal]),
                    ["proximal", "distal"],
                )
            return gripper_pb2.MotorCommandResponse(success=True)
        except Exception as e:
            logger.exception("Motor command failed")
            return gripper_pb2.MotorCommandResponse(success=False, error=str(e))

    def ReadMotors(self, request, context):
        with self._lock:
            pos1, pos2 = self._get_motor_positions()
        return gripper_pb2.MotorState(motor1_position=pos1, motor2_position=pos2)

    def SetTorque(self, request, context):
        # No-op in simulation (motors are always position-controlled)
        return gripper_pb2.TorqueResponse(success=True)

    def Ping(self, request, context):
        uptime = time.monotonic() - self._start_time
        return gripper_pb2.PingResponse(status="ok", uptime_seconds=uptime)
