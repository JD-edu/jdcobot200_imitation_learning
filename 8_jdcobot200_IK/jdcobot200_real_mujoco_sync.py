#!/usr/bin/env python3
"""Move the real JDCobot200 while mirroring encoder feedback in MuJoCo.

The target is checked by the same MINK IK and safe-zone pipeline used by
``jdcobot200_safe_mink_ik.py``.  After explicit confirmation, the real robot is
commanded with a minimum-jerk trajectory and the MuJoCo window displays the
latest measured encoder pose.  Press ESC to stop the trajectory and hold the
last measured pose.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from mujoco.glfw import glfw


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
SAFE_ROOT = REPO_ROOT / "7_FK_safe_zone"
HARDWARE_ROOT = REPO_ROOT / "9_robot_teaching_data_collection"
OFFSET_PATH = REPO_ROOT / "config" / "jdcobot200" / "offsets.txt"
LIMIT_PATH = REPO_ROOT / "1_jdcobot200_hardware_control" / "joint_limits.txt"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SAFE_ROOT))
sys.path.insert(0, str(HARDWARE_ROOT))

from jdcobot200_safe_mink_ik import (  # noqa: E402
    CENTER_TICK,
    DEFAULT_JOINT_SIGNS,
    MOTOR_IDS,
    check_safe_path,
    load_int_map,
    minimum_jerk,
    parse_signs,
    radians_to_ticks,
    read_positions,
    solve_position_ik,
    ticks_to_radians,
)
from motor_drive import MiniFeetechDriver  # noqa: E402
from safety_model import ARM_JOINT_NAMES, JDCobotSafetyModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="실제 JDCobot200을 이동하며 엔코더 자세를 MuJoCo에 표시합니다."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--target-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"),
        help="로봇 기준 절대 TCP 위치(m)",
    )
    target.add_argument(
        "--delta-xyz-mm", nargs=3, type=float, metavar=("DX", "DY", "DZ"),
        help="현재 TCP 기준 상대 이동(mm)",
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--move-time", type=float, default=3.0)
    parser.add_argument("--path-steps", type=int, default=201)
    parser.add_argument("--collision-margin-mm", type=float, default=5.0)
    parser.add_argument("--floor-margin-mm", type=float, default=10.0)
    parser.add_argument("--ik-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--ik-iterations", type=int, default=500)
    parser.add_argument("--max-joint-change-deg", type=float, default=10.0)
    parser.add_argument("--joint-signs", default=DEFAULT_JOINT_SIGNS)
    parser.add_argument(
        "--encoder-rate", type=float, default=20.0,
        help="엔코더 읽기 및 시뮬레이터 갱신 주파수(Hz, 기본 20)",
    )
    parser.add_argument(
        "--release-after-move", action="store_true",
        help="완료 또는 ESC 중단 후 토크 해제",
    )
    return parser.parse_args()


def create_window(model: mujoco.MjModel):
    if not glfw.init():
        raise RuntimeError("GLFW 초기화 실패: DISPLAY와 그래픽 드라이버를 확인하세요.")
    window = glfw.create_window(
        1200, 900, "JDCobot200 real robot + MuJoCo", None, None
    )
    if window is None:
        glfw.terminate()
        raise RuntimeError("GLFW 창 생성 실패")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    scene = mujoco.MjvScene(model, maxgeom=10_000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, 0.0, 0.22]
    camera.distance = 0.8
    camera.azimuth = 145.0
    camera.elevation = -25.0
    option = mujoco.MjvOption()

    def on_key(active_window, key, scancode, action, mods):
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(active_window, True)

    glfw.set_key_callback(window, on_key)
    return window, scene, context, camera, option


def render(window, safety: JDCobotSafetyModel, scene, context, camera, option,
           measured_q: np.ndarray, progress: float, error_mm: float) -> None:
    full_q = safety.configuration.q.copy()
    full_q[safety.qpos_addresses] = measured_q
    safety.configuration.update(full_q)
    width, height = glfw.get_framebuffer_size(window)
    mujoco.mjv_updateScene(
        safety.model, safety.data, option, None, camera,
        mujoco.mjtCatBit.mjCAT_ALL.value, scene,
    )
    mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), scene, context)
    glfw.set_window_title(
        window,
        f"JDCobot200 measured pose | progress={progress * 100:5.1f}% | "
        f"tracking={error_mm:5.2f} deg | ESC: STOP",
    )
    glfw.swap_buffers(window)
    glfw.poll_events()


def main() -> int:
    args = parse_args()
    if args.move_time <= 0 or args.path_steps < 2 or args.ik_iterations < 1:
        raise ValueError("move-time은 양수, path-steps는 2 이상이어야 합니다.")
    if args.encoder_rate <= 0 or args.max_joint_change_deg < 0:
        raise ValueError("encoder-rate는 양수, 최대 관절 변화량은 0 이상이어야 합니다.")
    if min(args.collision_margin_mm, args.floor_margin_mm,
           args.ik_tolerance_mm) < 0:
        raise ValueError("margin과 tolerance는 0 이상이어야 합니다.")

    safety = JDCobotSafetyModel()
    signs = parse_signs(args.joint_signs)
    offsets = load_int_map(OFFSET_PATH)
    limits = load_int_map(LIMIT_PATH, pair=True)
    centers = np.array([CENTER_TICK + int(offsets[int(i)]) for i in MOTOR_IDS])
    lower_ticks = np.array([
        limits[int(i)][0] + int(offsets[int(i)]) for i in MOTOR_IDS  # type: ignore[index]
    ], dtype=int)
    upper_ticks = np.array([
        limits[int(i)][1] + int(offsets[int(i)]) for i in MOTOR_IDS  # type: ignore[index]
    ], dtype=int)

    driver = None
    window = None
    torque_enabled = False
    try:
        driver = MiniFeetechDriver(args.port, args.baudrate)
        start_ticks = read_positions(driver)
        if np.any(start_ticks < lower_ticks) or np.any(start_ticks > upper_ticks):
            raise RuntimeError(f"현재 엔코더가 하드웨어 안전범위 밖입니다: {start_ticks}")
        start_q = ticks_to_radians(start_ticks, centers, signs)
        start_result = safety.evaluate(start_q)
        if not start_result.safe:
            raise RuntimeError(f"현재 자세가 safe-zone 밖입니다: {start_result.reasons}")

        target_xyz = (
            np.asarray(args.target_xyz, dtype=float)
            if args.target_xyz is not None
            else start_result.tcp_position
            + np.asarray(args.delta_xyz_mm, dtype=float) / 1000.0
        )
        goal_q, ik_error, iterations = solve_position_ik(
            safety, start_q, target_xyz, args.ik_iterations,
            args.ik_tolerance_mm / 1000.0,
        )
        # Position-only IK cannot observe rotation about the tool axis.  Letting
        # that null-space joint drift causes needless wrist motion, especially
        # around the vertical-arm singularity.  Keep wrist roll at its measured
        # starting angle and verify below that TCP position is still valid.
        goal_q[4] = start_q[4]
        goal_result = safety.evaluate(goal_q)
        ik_error = float(np.linalg.norm(goal_result.tcp_position - target_xyz))
        tolerance_m = args.ik_tolerance_mm / 1000.0
        if ik_error > tolerance_m:
            raise RuntimeError(f"IK가 허용 오차에 수렴하지 않았습니다: {ik_error * 1000:.2f} mm")

        joint_change = np.rad2deg(goal_q - start_q)
        if np.any(np.abs(joint_change) > args.max_joint_change_deg):
            required_limit = float(np.ceil(np.max(np.abs(joint_change))))
            raise RuntimeError(
                f"관절 변화 제한 {args.max_joint_change_deg:g} deg 초과: "
                f"{np.round(joint_change, 3)}\n"
                "현재 자세가 특이점에 가까우면 작은 TCP 이동도 base를 크게 "
                "회전시킬 수 있습니다. 먼저 시뮬레이션으로 경로를 확인한 뒤 "
                f"필요한 경우 --max-joint-change-deg {required_limit:g}와 더 긴 "
                "--move-time을 지정하세요."
            )
        goal_ticks = radians_to_ticks(goal_q, centers, signs)
        if np.any(goal_ticks < lower_ticks) or np.any(goal_ticks > upper_ticks):
            raise RuntimeError(f"IK 목표 tick이 하드웨어 제한 밖입니다: {goal_ticks}")

        collision_margin = args.collision_margin_mm / 1000.0
        floor_margin = args.floor_margin_mm / 1000.0
        unsafe, worst_self, worst_floor = check_safe_path(
            safety, start_q, goal_q, args.path_steps,
            collision_margin, floor_margin,
        )
        if unsafe is not None:
            index, q, reasons = unsafe
            raise RuntimeError(
                f"경로가 safe-zone을 벗어납니다: step={index}, "
                f"q(deg)={np.round(np.rad2deg(q), 3)}, reason={reasons}"
            )
        goal_result = safety.evaluate(goal_q, collision_margin, floor_margin)
        fk_error = float(np.linalg.norm(goal_result.tcp_position - target_xyz))
        if fk_error > tolerance_m:
            raise RuntimeError(f"시뮬레이션 FK 검증 실패: {fk_error * 1000:.2f} mm")

        print("\nMINK IK + SAFE-ZONE 사전 검사 통과")
        print("관절 순서:", ", ".join(ARM_JOINT_NAMES))
        print("현재 TCP(m):", np.round(start_result.tcp_position, 5))
        print("목표 TCP(m):", np.round(target_xyz, 5))
        print(f"IK/FK 오차: {fk_error * 1000:.3f} mm ({iterations} iterations)")
        print("관절 변화량(deg):", np.round(joint_change, 3))
        print(f"경로 최소 self/floor clearance: "
              f"{worst_self[0] * 1000:.2f}/{worst_floor[0] * 1000:.2f} mm")
        print("목표 tick:", goal_ticks)
        if input("시뮬레이터와 실제 로봇을 움직이려면 MOVE를 입력하세요: ").strip() != "MOVE":
            print("취소했습니다. 모터 명령을 보내지 않았습니다.")
            return 0

        window, scene, context, camera, option = create_window(safety.model)
        for motor_id, tick in zip(MOTOR_IDS, start_ticks):
            driver.set_position(int(motor_id), int(tick))
            if not driver.set_torque_verified(int(motor_id), True):
                raise RuntimeError(f"모터 ID {motor_id} Torque ON 확인 실패")
            time.sleep(0.05)
        torque_enabled = True

        started = time.perf_counter()
        next_command = started
        next_encoder = started
        measured_q = start_q.copy()
        last_sent = np.full(5, -1, dtype=int)
        stopped = False
        while True:
            now = time.perf_counter()
            progress = min(1.0, (now - started) / args.move_time)
            command_q = (1.0 - minimum_jerk(progress)) * start_q + minimum_jerk(progress) * goal_q

            if now >= next_command:
                command_ticks = radians_to_ticks(command_q, centers, signs)
                for index, (motor_id, tick) in enumerate(zip(MOTOR_IDS, command_ticks)):
                    if tick != last_sent[index]:
                        driver.set_position(int(motor_id), int(tick))
                        last_sent[index] = tick
                next_command = now + 0.025

            if now >= next_encoder:
                measured_ticks = read_positions(driver)
                measured_q = ticks_to_radians(measured_ticks, centers, signs)
                measured_result = safety.evaluate(
                    measured_q, collision_margin, floor_margin
                )
                if not measured_result.safe:
                    stopped = True
                    print(f"\n안전 중단: 실측 자세가 safe-zone 밖입니다: {measured_result.reasons}")
                next_encoder = now + 1.0 / args.encoder_rate

            tracking_error = float(np.max(np.abs(np.rad2deg(command_q - measured_q))))
            render(window, safety, scene, context, camera, option,
                   measured_q, progress, tracking_error)
            if glfw.window_should_close(window):
                stopped = progress < 1.0
                break
            if stopped or progress >= 1.0:
                break

        if stopped:
            hold_ticks = radians_to_ticks(measured_q, centers, signs)
            for motor_id, tick in zip(MOTOR_IDS, hold_ticks):
                driver.set_position(int(motor_id), int(tick))
            print("이동을 중단하고 마지막 실측 위치를 유지합니다.")
        else:
            time.sleep(0.3)
            measured_ticks = read_positions(driver)
            measured_q = ticks_to_radians(measured_ticks, centers, signs)
            measured_result = safety.evaluate(measured_q, collision_margin, floor_margin)
            real_fk_error = float(np.linalg.norm(measured_result.tcp_position - target_xyz))
            render(window, safety, scene, context, camera, option,
                   measured_q, 1.0, float(np.max(np.abs(np.rad2deg(goal_q - measured_q)))))
            print("\n동시 이동 완료")
            print("실측 q(deg):", np.round(np.rad2deg(measured_q), 3))
            print("실측 TCP(m):", np.round(measured_result.tcp_position, 5))
            print(f"실제 엔코더 FK 오차: {real_fk_error * 1000:.2f} mm")
        if not args.release_after_move:
            print("목표 자세 유지를 위해 토크를 켠 상태로 종료합니다.")
        return 2 if stopped else 0
    finally:
        if window is not None:
            glfw.destroy_window(window)
            glfw.terminate()
        if torque_enabled and args.release_after_move and driver is not None:
            for motor_id in MOTOR_IDS:
                try:
                    driver.set_torque_verified(int(motor_id), False)
                except Exception:
                    pass
        if driver is not None:
            driver.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
