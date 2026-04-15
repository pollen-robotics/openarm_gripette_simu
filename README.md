# OpenArm Gripette Simulation

MuJoCo simulation of the OpenArm right arm + Gripette gripper with gRPC control interfaces for policy testing.

## Install

```bash
uv sync
```

The robot model is fetched automatically from [pollen-robotics/openarm_gripette_model](https://github.com/pollen-robotics/openarm_gripette_model).

For local model development, edit `pyproject.toml` to use the local path source instead.

## Run the server

```bash
# With 3D viewer
uv run python -m openarm_gripette_simu

# With a specific scene
uv run python -m openarm_gripette_simu --scene scenes/table_red_cube.xml

# Headless (no viewer)
uv run python -m openarm_gripette_simu --headless
```

The server exposes two gRPC services on separate ports:

| Service | Port | Description |
|---------|------|-------------|
| GripperService | 50051 | Camera stream + gripper motor control (same API as real Gripette) |
| ArmService | 50052 | Delta Cartesian arm control |

## gRPC API

### GripperService (port 50051)

Identical to the real [Gripette](https://github.com/pollen-robotics/gripette) gRPC API. Existing clients work unchanged.

| RPC | Type | Description |
|-----|------|-------------|
| `StreamState` | server-stream | 10Hz JPEG camera frames + motor positions (rad) + timestamp |
| `SendMotorCommand(motor1_goal, motor2_goal)` | unary | Set gripper joint goals (rad) |
| `ReadMotors` | unary | Read gripper joint positions (rad) |
| `SetTorque(enable)` | unary | No-op in simulation |
| `Ping` | unary | Health check + uptime |

### ArmService (port 50052)

| RPC | Type | Description |
|-----|------|-------------|
| `SendCartesianDelta(dx, dy, dz, dr6d[6])` | unary | Apply a delta to the end-effector pose (meters + 6D rotation) |
| `GetArmState` | unary | Current pose (xyz + r6d) and 7 arm joint positions (rad) |
| `Ping` | unary | Health check + uptime |

The 6D rotation representation follows Zhou et al. (CVPR 2019): first two columns of the rotation matrix, 6 values. This matches the action space used by diffusion policies trained with LeRobot.

### Quick client example

```python
import grpc
from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc

stub = arm_pb2_grpc.ArmServiceStub(grpc.insecure_channel("localhost:50052"))

# Read current state
state = stub.GetArmState(arm_pb2.GetArmStateRequest())
print(f"EE position: [{state.x:.3f}, {state.y:.3f}, {state.z:.3f}]")

# Move 1cm in x
stub.SendCartesianDelta(arm_pb2.CartesianDelta(
    dx=0.01, dy=0.0, dz=0.0, dr6d=[0.0] * 6,
))
```

### View the camera stream

```bash
uv run python -c "
import cv2, grpc, numpy as np
from openarm_gripette_simu.proto import gripper_pb2, gripper_pb2_grpc
stub = gripper_pb2_grpc.GripperServiceStub(grpc.insecure_channel('localhost:50051'))
for frame in stub.StreamState(gripper_pb2.StreamRequest()):
    img = cv2.imdecode(np.frombuffer(frame.jpeg_data, np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow('Gripette', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
"
```

## Scenes

Scene XML files live in `scenes/`. They include the robot model and add environment elements.

| Scene | Description |
|-------|-------------|
| `scenes/table_red_cube.xml` | Wooden table with a movable red cube |

To create a new scene, copy `table_red_cube.xml` as a template. The robot is included via `<include file="...robot.xml"/>`.

## Examples

Standalone demos (no gRPC, direct simulation control):

```bash
uv run python examples/cartesian_square.py   # Cartesian square trajectory
uv run python examples/joint_control.py      # Move joints one by one
uv run python examples/gripper_demo.py       # Gripper open/close cycle
uv run python examples/table_scene.py        # Table scene with red cube
```

## Camera

The simulated Gripette camera matches the real camera calibration:
- Resolution: 1296x972
- Lens model: KannalaBrandt8 fisheye
- MuJoCo renders a wide-FOV pinhole image (130°), then remaps it with the real distortion coefficients

## Regenerating proto stubs

```bash
uv sync --extra dev
uv run python -m grpc_tools.protoc -I proto \
    --python_out=openarm_gripette_simu/proto \
    --grpc_python_out=openarm_gripette_simu/proto \
    --pyi_out=openarm_gripette_simu/proto \
    proto/gripper.proto proto/arm.proto
```

Then fix the imports in the generated `*_pb2_grpc.py` files: change `import xxx_pb2` to `from . import xxx_pb2`.
