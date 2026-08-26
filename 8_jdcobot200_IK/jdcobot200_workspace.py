#!/usr/bin/env python3
"""Sample the JDCobot200 kinematic workspace with the MuJoCo model.

This computes positions reachable by ``graspframe`` within the arm actuator
limits.  It does not reject self-collisions because collision is disabled in
the current XML model.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_XML = ROOT / "scene.xml"
ARM_JOINT_NAMES = ("base", "shoulder", "elbow", "wrist_pitch", "wrist_roll")
END_EFFECTOR_SITE = "graspframe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JDCobot200 graspframe 작업공간을 MuJoCo FK로 샘플링합니다."
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "jdcobot200_workspace.npz"
    )
    parser.add_argument(
        "--csv", type=Path, default=None, help="지정하면 모든 샘플을 CSV로도 저장"
    )
    parser.add_argument(
        "--plot", type=Path, default=ROOT / "jdcobot200_workspace.png"
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def require_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, kind, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo model does not contain {kind.name} '{name}'.")
    return int(object_id)


def get_arm_limits(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arm qpos addresses and effective lower/upper command limits."""
    qpos_addresses = []
    limits = []
    for name in ARM_JOINT_NAMES:
        joint_id = require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))

        joint_range = model.jnt_range[joint_id]
        actuator_range = model.actuator_ctrlrange[actuator_id]
        limits.append(
            [
                max(joint_range[0], actuator_range[0]),
                min(joint_range[1], actuator_range[1]),
            ]
        )

    limits_array = np.asarray(limits)
    if np.any(limits_array[:, 0] >= limits_array[:, 1]):
        raise ValueError("Joint and actuator ranges have an empty intersection.")
    return (
        np.asarray(qpos_addresses, dtype=int),
        limits_array[:, 0],
        limits_array[:, 1],
    )


def sample_workspace(
    model: mujoco.MjModel, sample_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if sample_count < 1:
        raise ValueError("--samples must be at least 1.")

    qpos_addresses, lower, upper = get_arm_limits(model)
    site_id = require_id(model, mujoco.mjtObj.mjOBJ_SITE, END_EFFECTOR_SITE)
    rng = np.random.default_rng(seed)

    # Include the home pose and all joint-limit corners, then Monte Carlo
    # samples. Corners make simple axis bounds less dependent on random luck.
    corners = np.array(
        [
            [upper[j] if mask & (1 << j) else lower[j] for j in range(5)]
            for mask in range(1 << 5)
        ],
        dtype=float,
    )
    random_count = max(0, sample_count - len(corners) - 1)
    random_q = rng.uniform(lower, upper, size=(random_count, len(lower)))
    joint_samples = np.vstack([np.zeros((1, 5)), corners, random_q])[:sample_count]

    data = mujoco.MjData(model)
    positions = np.empty((len(joint_samples), 3), dtype=float)
    for index, joint_values in enumerate(joint_samples):
        data.qpos[qpos_addresses] = joint_values
        mujoco.mj_forward(model, data)
        positions[index] = data.site_xpos[site_id]

    return positions, joint_samples, lower, upper


def save_csv(path: Path, positions: np.ndarray, joint_samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow((*ARM_JOINT_NAMES, "x", "y", "z"))
        writer.writerows(np.hstack([joint_samples, positions]))


def save_plot(path: Path, positions: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Use --no-plot or install matplotlib."
        ) from exc

    # Rendering every point is expensive and visually dense.
    stride = max(1, len(positions) // 30_000)
    points = positions[::stride]
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.18)
    axis.scatter(*positions[0], color="red", s=45, label="home graspframe")
    axis.set(xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]")
    axis.set_title(f"JDCobot200 kinematic workspace ({len(positions):,} samples)")
    axis.set_box_aspect(np.maximum(np.ptp(positions, axis=0), 1e-6))
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    positions, joint_samples, lower, upper = sample_workspace(
        model, args.samples, args.seed
    )

    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)
    radius = np.linalg.norm(positions[:, :2], axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        positions=positions,
        joint_positions=joint_samples,
        joint_names=np.asarray(ARM_JOINT_NAMES),
        joint_lower=lower,
        joint_upper=upper,
    )
    if args.csv is not None:
        save_csv(args.csv, positions, joint_samples)
    if not args.no_plot:
        save_plot(args.plot, positions)

    print(f"샘플 수: {len(positions):,}")
    for axis_name, lo, hi in zip("XYZ", minimum, maximum):
        print(f"{axis_name} 범위: {lo:+.4f} ~ {hi:+.4f} m  (폭 {(hi-lo):.4f} m)")
    print(f"수평 최대 반경: {radius.max():.4f} m")
    print(f"NPZ 저장: {args.output.resolve()}")
    if args.csv is not None:
        print(f"CSV 저장: {args.csv.resolve()}")
    if not args.no_plot:
        print(f"그림 저장: {args.plot.resolve()}")


if __name__ == "__main__":
    main()
