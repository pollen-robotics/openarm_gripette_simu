"""Fisheye camera model (KannalaBrandt8) for the Gripette camera.

Renders a wide-FOV pinhole image from MuJoCo, then remaps it to simulate
the real fisheye camera with matching intrinsics and distortion.
"""

import numpy as np
import cv2

# Real camera calibration (from rpi_bmi088_slam_settings.yaml)
CAMERA_WIDTH = 1296
CAMERA_HEIGHT = 972
CAMERA_FX = 537.8970067668539
CAMERA_FY = 536.2166303906646
CAMERA_CX = 650.3978870138183
CAMERA_CY = 511.9259906764862
CAMERA_K1 = -0.04508959072032921
CAMERA_K2 = 0.07499821258630092
CAMERA_K3 = -0.2048230788391512
CAMERA_K4 = 0.16387495554310924

# MuJoCo pinhole render settings
# fovy=130° covers 74.4° diagonal — enough for 70.5° max corner angle
PINHOLE_FOVY = 130.0
# Render at 2x resolution for quality (pinhole stretches edges)
PINHOLE_RENDER_SCALE = 2


def _kb8_inverse_theta(r_d: np.ndarray, k1: float, k2: float, k3: float, k4: float, n_iter: int = 20) -> np.ndarray:
    """Solve KannalaBrandt8 inverse: find theta given r_d = theta_d.

    theta_d = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9
    Uses Newton's method.
    """
    theta = r_d.copy()
    for _ in range(n_iter):
        t2 = theta * theta
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4
        f = theta * (1 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8) - r_d
        fp = 1 + 3 * k1 * t2 + 5 * k2 * t4 + 7 * k3 * t6 + 9 * k4 * t8
        theta = theta - f / fp
    return theta


def build_fisheye_remap(
    out_width: int = CAMERA_WIDTH,
    out_height: int = CAMERA_HEIGHT,
    pinhole_width: int | None = None,
    pinhole_height: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the remap tables from fisheye pixels to pinhole pixels.

    For each pixel (u, v) in the output fisheye image:
      1. Unproject via KannalaBrandt8 inverse to get a 3D ray
      2. Project that ray into the pinhole image

    Returns (map_x, map_y) for use with cv2.remap().
    """
    if pinhole_width is None:
        pinhole_width = out_width * PINHOLE_RENDER_SCALE
    if pinhole_height is None:
        pinhole_height = out_height * PINHOLE_RENDER_SCALE

    # Pinhole intrinsics (derived from MuJoCo fovy)
    half_fovy = np.radians(PINHOLE_FOVY / 2)
    fy_pin = (pinhole_height / 2) / np.tan(half_fovy)
    fx_pin = fy_pin  # square pixels
    cx_pin = pinhole_width / 2
    cy_pin = pinhole_height / 2

    # Grid of output fisheye pixel coordinates
    u_fish, v_fish = np.meshgrid(
        np.arange(out_width, dtype=np.float32),
        np.arange(out_height, dtype=np.float32),
    )

    # Normalized fisheye coordinates
    mx = (u_fish - CAMERA_CX) / CAMERA_FX
    my = (v_fish - CAMERA_CY) / CAMERA_FY
    r_d = np.sqrt(mx ** 2 + my ** 2)

    # Solve for theta (angle from optical axis)
    theta = np.zeros_like(r_d)
    mask = r_d > 1e-8
    theta[mask] = _kb8_inverse_theta(r_d[mask], CAMERA_K1, CAMERA_K2, CAMERA_K3, CAMERA_K4)

    # Convert to pinhole normalized coordinates: (tan(theta)/r_d) * [mx, my]
    scale = np.ones_like(r_d)
    scale[mask] = np.tan(theta[mask]) / r_d[mask]

    pin_x = scale * mx
    pin_y = scale * my

    # Project to pinhole pixel coordinates
    map_x = (fx_pin * pin_x + cx_pin).astype(np.float32)
    map_y = (fy_pin * pin_y + cy_pin).astype(np.float32)

    return map_x, map_y


class FisheyeCamera:
    """Applies fisheye distortion to MuJoCo pinhole renders.

    Usage:
        cam = FisheyeCamera()
        # In MuJoCo, set gripette_cam fovy to cam.pinhole_fovy
        # Render at cam.pinhole_width x cam.pinhole_height
        pinhole_img = renderer.render()
        fisheye_img = cam.distort(pinhole_img)
    """

    def __init__(self, render_scale: int = PINHOLE_RENDER_SCALE):
        self.out_width = CAMERA_WIDTH
        self.out_height = CAMERA_HEIGHT
        self.pinhole_width = CAMERA_WIDTH * render_scale
        self.pinhole_height = CAMERA_HEIGHT * render_scale
        self.pinhole_fovy = PINHOLE_FOVY

        # Precompute remap tables
        self._map_x, self._map_y = build_fisheye_remap(
            self.out_width, self.out_height,
            self.pinhole_width, self.pinhole_height,
        )

    def distort(self, pinhole_img: np.ndarray) -> np.ndarray:
        """Apply fisheye distortion to a pinhole-rendered image.

        Args:
            pinhole_img: RGB image from MuJoCo (pinhole_height x pinhole_width x 3).

        Returns:
            Fisheye image (out_height x out_width x 3).
        """
        return cv2.remap(pinhole_img, self._map_x, self._map_y, cv2.INTER_LINEAR, borderValue=0)
