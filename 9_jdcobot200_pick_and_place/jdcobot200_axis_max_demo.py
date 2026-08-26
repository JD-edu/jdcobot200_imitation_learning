#!/usr/bin/env python3
"""Move JDCobot200 through the maximum reachable X, Y, and Z poses."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from mujoco.glfw import glfw
import mujoco
import numpy as np

from jdcobot200_workspace import (
    ARM_JOINT_NAMES,
    END_EFFECTOR_SITE,
    get_arm_limits,
    sample_workspace,
)


ROOT = Path(__file__).resolve().parent
XML_PATH = ROOT / "scene.xml"
GRIPPER_CLOSED = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="엔드이펙터의 X/Y/Z 최대 도달 자세를 순차 재생합니다."
    )
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--move-time", type=float, default=3.0)
    parser.add_argument("--hold-time", type=float, default=1.5)
    parser.add_argument("--once", action="store_true", help="한 사이클 후 종료")
    return parser.parse_args()


def refine_axis_maximum(
    model: mujoco.MjModel,
    seed_q: np.ndarray,
    axis: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Polish a sampled maximum with bounded numerical optimization."""
    qpos_addresses, _, _ = get_arm_limits(model)
    site_id = model.site(END_EFFECTOR_SITE).id
    data = mujoco.MjData(model)

    def objective(q: np.ndarray) -> float:
        data.qpos[qpos_addresses] = q
        mujoco.mj_forward(model, data)
        return -float(data.site_xpos[site_id, axis])

    best_q = seed_q.copy()
    try:
        from scipy.optimize import minimize

        result = minimize(
            objective,
            best_q,
            method="L-BFGS-B",
            bounds=list(zip(lower, upper)),
            options={"maxiter": 300, "ftol": 1e-12},
        )
        if result.success or result.fun < objective(best_q):
            best_q = np.clip(result.x, lower, upper)
    except ImportError:
        pass

    # graspframe lies on the wrist-roll axis, so roll does not increase its
    # reachable position. Keep it neutral for a clearer, repeatable demo.
    best_q[ARM_JOINT_NAMES.index("wrist_roll")] = 0.0
    data.qpos[qpos_addresses] = best_q
    mujoco.mj_forward(model, data)
    return best_q, data.site_xpos[site_id].copy()


def find_maximum_poses(
    model: mujoco.MjModel, samples: int, seed: int
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    positions, joint_samples, lower, upper = sample_workspace(model, samples, seed)
    poses = []
    for axis, label in enumerate(("X MAX", "Y MAX", "Z MAX")):
        sample_index = int(np.argmax(positions[:, axis]))
        q, xyz = refine_axis_maximum(
            model, joint_samples[sample_index], axis, lower, upper
        )
        poses.append((label, q, xyz))
    return poses


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def main() -> None:
    args = parse_args()
    if args.move_time <= 0 or args.hold_time < 0:
        raise ValueError("--move-time은 양수, --hold-time은 0 이상이어야 합니다.")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    arm_qpos_addresses, lower, upper = get_arm_limits(model)
    arm_actuator_ids = np.array([model.actuator(name).id for name in ARM_JOINT_NAMES])
    gripper_actuator_id = model.actuator("gripper_motor").id
    site_id = model.site(END_EFFECTOR_SITE).id

    print(f"{args.samples:,}개 자세로 X/Y/Z 최대점을 탐색합니다...")
    maximum_poses = find_maximum_poses(model, args.samples, args.seed)
    for label, q, xyz in maximum_poses:
        print(f"{label}: xyz={np.round(xyz, 5)} m, q={np.round(q, 4)} rad")

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    home_q = data.qpos[arm_qpos_addresses].copy()
    home_xyz = data.site_xpos[site_id].copy()
    sequence = [("HOME", home_q, home_xyz), *maximum_poses]

    data.ctrl[arm_actuator_ids] = home_q
    data.ctrl[gripper_actuator_id] = GRIPPER_CLOSED

    if not glfw.init():
        raise RuntimeError("GLFW initialization failed.")
    window = glfw.create_window(1200, 900, "JDCobot200 axis maxima", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("GLFW window creation failed.")

    glfw.make_context_current(window)
    glfw.swap_interval(1)
    scene = mujoco.MjvScene(model, maxgeom=10_000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    mujoco.mjv_defaultCamera(camera)
    mujoco.mjv_defaultOption(option)
    camera.lookat[:] = [0.0, 0.0, 0.2]
    camera.distance = 1.25
    camera.azimuth = 145
    camera.elevation = -22

    pose_index = 0
    start_q = home_q.copy()
    phase_start = time.perf_counter()
    holding = False
    completed_cycles = 0

    try:
        while not glfw.window_should_close(window):
            now = time.perf_counter()
            label, target_q, target_xyz = sequence[pose_index]

            if not holding:
                progress = (now - phase_start) / args.move_time
                blend = smoothstep(progress)
                command_q = (1.0 - blend) * start_q + blend * target_q
                data.ctrl[arm_actuator_ids] = np.clip(command_q, lower, upper)
                if progress >= 1.0:
                    holding = True
                    phase_start = now
            elif now - phase_start >= args.hold_time:
                start_q = data.ctrl[arm_actuator_ids].copy()
                pose_index += 1
                if pose_index >= len(sequence):
                    pose_index = 0
                    completed_cycles += 1
                    if args.once:
                        break
                holding = False
                phase_start = now

            data.ctrl[gripper_actuator_id] = GRIPPER_CLOSED
            frame_start = data.time
            while data.time - frame_start < 1.0 / 60.0:
                mujoco.mj_step(model, data)

            actual_xyz = data.site_xpos[site_id]
            glfw.set_window_title(
                window,
                f"JDCobot200 | {label} | "
                f"target=({target_xyz[0]:+.3f}, {target_xyz[1]:+.3f}, {target_xyz[2]:+.3f}) | "
                f"actual=({actual_xyz[0]:+.3f}, {actual_xyz[1]:+.3f}, {actual_xyz[2]:+.3f})",
            )
            width, height = glfw.get_framebuffer_size(window)
            mujoco.mjv_updateScene(
                model,
                data,
                option,
                None,
                camera,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                scene,
            )
            mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), scene, context)
            glfw.swap_buffers(window)
            glfw.poll_events()
    finally:
        context.free()
        scene.free()
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
