#!/usr/bin/env python3
"""XYZ 목표의 도달성·안전성을 검사하고 MuJoCo에서 로봇팔을 이동합니다."""

# 절대 TCP 목표(m):
# /usr/bin/python3 projects/curriculum/05_mujoco_ik/02_ik_mujoco_simulation.py \
#   --target-xyz 0.05 0.10 0.35 --start-q-deg 0 20 -40 20 0
#
# 현재 TCP 기준 상대 목표(mm):
# /usr/bin/python3 projects/curriculum/05_mujoco_ik/02_ik_mujoco_simulation.py \
#   --delta-xyz-mm 20 20 -5 --start-q-deg 0 0 0 0 0 --move-time 4

from __future__ import annotations

import argparse
import sys
import time

import mujoco
from mujoco.glfw import glfw
import numpy as np

from ik_solver import SafeIK, print_solution


def smoothstep(progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress * progress * (3.0 - 2.0 * progress)


def simulate(solver: SafeIK, start_q: np.ndarray, goal_q: np.ndarray, *,
             move_time: float, hold_time: float) -> tuple[np.ndarray, bool]:
    model = solver.model
    data = mujoco.MjData(model)
    # Keep the canonical order from the safety model rather than duplicating it.
    from ik_solver import JOINT_NAMES
    actuator_ids = np.asarray([model.actuator(name).id for name in JOINT_NAMES])
    qpos_addresses = solver.fk.qpos_addresses

    data.qpos[qpos_addresses] = start_q
    data.ctrl[actuator_ids] = start_q
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_motor")
    if gripper_id >= 0:
        data.ctrl[gripper_id] = 0.0
    mujoco.mj_forward(model, data)

    if not glfw.init():
        raise RuntimeError("GLFW 초기화 실패: 데스크톱 DISPLAY에서 실행하세요.")
    window = glfw.create_window(900, 700, "JDCobot200 IK MuJoCo simulation", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("GLFW 창 생성 실패")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    camera, option = mujoco.MjvCamera(), mujoco.MjvOption()
    camera.lookat[:], camera.distance = [0.12, 0.0, 0.20], 0.75
    camera.azimuth, camera.elevation = 145.0, -25.0
    scene = mujoco.MjvScene(model, maxgeom=10_000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    def on_key(_window, key, _scancode, action, _mods):
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(_window, True)

    glfw.set_key_callback(window, on_key)
    started = time.perf_counter()
    completed = False
    try:
        while not glfw.window_should_close(window):
            elapsed = time.perf_counter() - started
            progress = min(1.0, elapsed / move_time)
            alpha = smoothstep(progress)
            command_q = (1.0 - alpha) * start_q + alpha * goal_q
            data.ctrl[actuator_ids] = command_q

            frame_end = data.time + 1.0 / 60.0
            while data.time < frame_end:
                mujoco.mj_step(model, data)

            actual_q = data.qpos[qpos_addresses].copy()
            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(
                model, data, option, None, camera,
                mujoco.mjtCatBit.mjCAT_ALL.value, scene,
            )
            mujoco.mjr_render(viewport, scene, context)
            glfw.set_window_title(
                window,
                f"IK MuJoCo | {progress*100:5.1f}% | "
                f"q={np.round(np.rad2deg(actual_q), 1)} | ESC: stop",
            )
            glfw.swap_buffers(window)
            glfw.poll_events()
            if progress >= 1.0 and elapsed >= move_time + hold_time:
                completed = True
                break
            time.sleep(0.001)
    finally:
        final_q = data.qpos[qpos_addresses].copy()
        glfw.destroy_window(window)
        glfw.terminate()
    return final_q, completed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="XYZ 목표를 검사한 뒤 MuJoCo position actuator로 이동을 시뮬레이션합니다."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="base 좌표계 절대 TCP 위치(m)")
    target.add_argument("--delta-xyz-mm", nargs=3, type=float, metavar=("DX", "DY", "DZ"),
                        help="시작 TCP 기준 상대 이동(mm)")
    parser.add_argument("--start-q-deg", nargs=5, type=float, default=[0] * 5)
    parser.add_argument("--tolerance-mm", type=float, default=1.0)
    parser.add_argument("--path-steps", type=int, default=301)
    parser.add_argument("--move-time", type=float, default=4.0)
    parser.add_argument("--hold-time", type=float, default=1.0)
    args = parser.parse_args()
    if args.move_time <= 0 or args.hold_time < 0 or args.path_steps < 2:
        raise ValueError("move-time은 양수, hold-time은 0 이상, path-steps는 2 이상이어야 합니다.")

    solver = SafeIK()
    start_q = np.deg2rad(np.asarray(args.start_q_deg, dtype=float))
    start = solver.fk.evaluate(start_q)
    if not start.safe:
        raise RuntimeError(f"시작 자세가 안전하지 않습니다: {start.reasons}")
    target_xyz = (
        np.asarray(args.target_xyz, dtype=float) if args.target_xyz is not None
        else start.tcp_position + np.asarray(args.delta_xyz_mm, dtype=float) / 1000.0
    )

    # solve() 내부에서 최대 도달 반경을 먼저 검사합니다.
    result = solver.solve(start_q, target_xyz, tolerance_m=args.tolerance_mm / 1000.0)
    if not result.converged:
        raise RuntimeError(f"IK 미수렴: 최종 FK 오차 {result.error_m*1000:.2f} mm")
    path = solver.check_path(start_q, result.q_rad, steps=args.path_steps)
    print_solution(start_q, target_xyz, result, path)
    if not path.safe:
        raise RuntimeError(
            f"경로가 안전하지 않습니다: step={path.unsafe_step}, "
            f"reason={path.unsafe_result.reasons}"
        )
    print("REACHABLE + SAFE: MuJoCo 시뮬레이션을 시작합니다. ESC로 중단할 수 있습니다.")

    final_q, completed = simulate(
        solver, start_q, result.q_rad,
        move_time=args.move_time, hold_time=args.hold_time,
    )
    final_pose = solver.fk.evaluate(final_q)
    final_error_mm = float(np.linalg.norm(final_pose.tcp_position - target_xyz) * 1000.0)
    print("시뮬레이션 최종 q(deg):", np.round(np.rad2deg(final_q), 3))
    print("시뮬레이션 최종 TCP(m):", np.round(final_pose.tcp_position, 6))
    print(f"position actuator 최종 오차: {final_error_mm:.3f} mm")
    if not completed:
        print("사용자가 시뮬레이션을 중단했습니다.")
        return 130
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
