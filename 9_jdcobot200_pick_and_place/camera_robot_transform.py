#!/usr/bin/env python3
"""Apply a saved Astra-camera to JDCobot200 IK-base transformation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load_transform(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(payload["T_base_camera"], dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("T_base_camera must be a finite 4x4 matrix")
    return matrix


def camera_to_base(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.shape[-1] != 3:
        raise ValueError("points must end in x,y,z")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def depth_pixel_to_camera(u: float, v: float, depth_mm: float,
                          camera_matrix: np.ndarray) -> np.ndarray:
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    z = depth_mm / 1000.0
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])


def main() -> None:
    parser = argparse.ArgumentParser(description="Camera XYZ to JDCobot IK XYZ")
    parser.add_argument("x", type=float, help="camera X in metres")
    parser.add_argument("y", type=float, help="camera Y in metres")
    parser.add_argument("z", type=float, help="camera Z in metres")
    parser.add_argument("--calibration", type=Path,
                        default=ROOT / "camera_to_robot_calibration.json")
    args = parser.parse_args()
    result = camera_to_base(np.array([args.x, args.y, args.z]),
                            load_transform(args.calibration))
    print("IK/base XYZ [m]:", " ".join(f"{value:.7f}" for value in result))


if __name__ == "__main__":
    main()
