#!/usr/bin/env python3
"""Joint-space READY approach followed by singularity-checked Cartesian IK.

Sequence:
  measured pose -> READY (joint space) -> Cartesian IK target -> optional return

The MuJoCo window mirrors measured encoder positions throughout the motion.
HOME remains the calibration/storage pose and is never used as an IK start pose.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from mujoco.glfw import glfw
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
SAFE_ROOT = REPO_ROOT / "7_FK_safe_zone"
HARDWARE_ROOT = REPO_ROOT / "9_robot_teaching_data_collection"
OFFSET_PATH = REPO_ROOT / "config" / "jdcobot200" / "offsets.txt"
LIMIT_PATH = REPO_ROOT / "1_jdcobot200_hardware_control" / "joint_limits.txt"

sys.path[:0] = [str(ROOT), str(SAFE_ROOT), str(HARDWARE_ROOT)]

from jdcobot200_real_mujoco_sync import create_window, render  # noqa: E402
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


# Use the negative-elbow IK branch for the usual +X/+Y work area.  Moving from
# HOME to READY is deliberately joint-space control, so passing the HOME
# singularity during startup does not ask Cartesian IK to choose a branch.
DEFAULT_READY_DEG = (0.0, 20.0, -40.0, 20.0, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="READY 자세에서 특이점을 검사한 뒤 Cartesian IK를 실행합니다."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--target-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"),
        help="로봇 기준 절대 TCP 위치(m)",
    )
    target.add_argument(
        "--delta-xyz-mm", nargs=3, type=float, metavar=("DX", "DY", "DZ"),
        help="READY TCP 기준 상대 이동(mm)",
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument(
        "--ready-q-deg", nargs=5, type=float, default=DEFAULT_READY_DEG,
        metavar=("J1", "J2", "J3", "J4", "J5"),
        help="IK 시작 READY 관절각(기본: 0 20 -40 20 0 deg)",
    )
    parser.add_argument("--ready-time", type=float, default=5.0)
    parser.add_argument("--move-time", type=float, default=3.0)
    parser.add_argument(
        "--goal-hold-time", type=float, default=3.0,
        help="IK 목표 도착 후 자세를 표시하는 시간(초, 기본 3)",
    )
    parser.add_argument("--return-time", type=float, default=5.0)
    parser.add_argument(
        "--return-to", choices=("stay", "ready", "home"), default="stay",
        help="작업 후 이동 위치(기본: stay)",
    )
    parser.add_argument("--path-steps", type=int, default=201)
    parser.add_argument("--collision-margin-mm", type=float, default=5.0)
    parser.add_argument("--floor-margin-mm", type=float, default=10.0)
    parser.add_argument("--ik-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--ik-iterations", type=int, default=500)
    parser.add_argument(
        "--max-ik-joint-change-deg", type=float,
        help="READY부터 목표까지 관절별 최대 변화(기본: 상대 10, 절대 120 deg)",
    )
    parser.add_argument(
        "--ik-restarts", type=int, default=64,
        help="절대 좌표 IK의 다중 초기값 탐색 수(기본 64)",
    )
    parser.add_argument(
        "--min-jacobian-singular-value", type=float, default=0.02,
        help="Cartesian 경로 최소 허용 Jacobian 특이값(기본 0.02)",
    )
    parser.add_argument("--encoder-rate", type=float, default=20.0)
    parser.add_argument("--joint-signs", default=DEFAULT_JOINT_SIGNS)
    parser.add_argument("--release-after-move", action="store_true")
    return parser.parse_args()


def jacobian_metrics(safety: JDCobotSafetyModel, q: np.ndarray) -> tuple[float, float]:
    """Return translational Jacobian minimum singular value and condition."""
    safety.evaluate(q, collision_margin=0.0, floor_margin=0.0)
    jacobian = np.zeros((3, safety.model.nv), dtype=float)
    rotational = np.zeros((3, safety.model.nv), dtype=float)
    mujoco.mj_jacSite(
        safety.model, safety.data, jacobian, rotational, safety.site_id
    )
    dofs = safety.model.jnt_dofadr[safety.joint_ids].astype(int)
    singular_values = np.linalg.svd(jacobian[:, dofs], compute_uv=False)
    minimum = float(singular_values[-1])
    condition = (
        float(singular_values[0] / minimum) if minimum > 1e-12 else float("inf")
    )
    return minimum, condition


def check_singularity_path(
    safety: JDCobotSafetyModel, start_q: np.ndarray, goal_q: np.ndarray,
    steps: int, minimum_allowed: float,
) -> tuple[float, float, int]:
    worst = (float("inf"), 0.0, 0)
    for index, q in enumerate(np.linspace(start_q, goal_q, steps)):
        minimum, condition = jacobian_metrics(safety, q)
        if minimum < worst[0]:
            worst = (minimum, condition, index)
    if worst[0] < minimum_allowed:
        raise RuntimeError(
            "Cartesian 경로가 특이점에 너무 가깝습니다: "
            f"step={worst[2]}, sigma_min={worst[0]:.6f}, "
            f"condition={worst[1]:.1f}, limit={minimum_allowed:.6f}"
        )
    return worst


def check_joint_path(
    safety: JDCobotSafetyModel, start_q: np.ndarray, goal_q: np.ndarray,
    args: argparse.Namespace, label: str,
) -> tuple:
    unsafe, worst_self, worst_floor = check_safe_path(
        safety, start_q, goal_q, args.path_steps,
        args.collision_margin_mm / 1000.0,
        args.floor_margin_mm / 1000.0,
    )
    if unsafe is not None:
        index, q, reasons = unsafe
        raise RuntimeError(
            f"{label} 경로가 safe-zone을 벗어납니다: step={index}, "
            f"q(deg)={np.round(np.rad2deg(q), 3)}, reason={reasons}"
        )
    return worst_self, worst_floor


def solve_absolute_position_ik(
    safety: JDCobotSafetyModel, ready_q: np.ndarray, target_xyz: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, int]:
    """Find a safe absolute-position solution using deterministic restarts."""
    rng = np.random.default_rng(20260904)
    seeds = [ready_q[:4].copy()]
    seeds.extend(
        rng.uniform(safety.lower[:4], safety.upper[:4])
        for _ in range(args.ik_restarts - 1)
    )
    candidates = []

    def position_error(q4: np.ndarray) -> np.ndarray:
        q = np.r_[q4, ready_q[4]]
        return (safety.evaluate(q, 0.0, 0.0).tcp_position - target_xyz) * 1000.0

    for seed in seeds:
        result = least_squares(
            position_error, seed,
            bounds=(safety.lower[:4], safety.upper[:4]),
            max_nfev=max(500, args.ik_iterations),
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )
        q = np.r_[result.x, ready_q[4]]
        error = float(np.linalg.norm(position_error(result.x)))
        if error > args.ik_tolerance_mm:
            continue
        unsafe, _, _ = check_safe_path(
            safety, ready_q, q, args.path_steps,
            args.collision_margin_mm / 1000.0,
            args.floor_margin_mm / 1000.0,
        )
        if unsafe is not None:
            continue
        try:
            check_singularity_path(
                safety, ready_q, q, args.path_steps,
                args.min_jacobian_singular_value,
            )
        except RuntimeError:
            continue
        distance = float(np.linalg.norm((q - ready_q) / (safety.upper - safety.lower)))
        candidates.append((distance, error, q))
    if not candidates:
        raise RuntimeError(
            f"{args.ik_restarts}개 초기값에서 허용 오차·safe-zone·특이점 "
            "조건을 모두 만족하는 절대 좌표 IK 해를 찾지 못했습니다."
        )
    _, error_mm, solution = min(candidates, key=lambda item: item[0])
    return solution, error_mm / 1000.0, len(candidates)


def validate_ticks(q: np.ndarray, centers: np.ndarray, signs: np.ndarray,
                   lower: np.ndarray, upper: np.ndarray, label: str) -> np.ndarray:
    ticks = radians_to_ticks(q, centers, signs)
    if np.any(ticks < lower) or np.any(ticks > upper):
        raise RuntimeError(f"{label} tick이 하드웨어 제한 밖입니다: {ticks}")
    return ticks


def move_segment(
    driver: MiniFeetechDriver, safety: JDCobotSafetyModel, window,
    scene, context, camera, option, start_q: np.ndarray, goal_q: np.ndarray,
    duration: float, centers: np.ndarray, signs: np.ndarray,
    encoder_rate: float, collision_margin: float, floor_margin: float,
    label: str,
) -> tuple[np.ndarray, bool]:
    """Command one joint-space segment and display measured encoder motion."""
    started = time.perf_counter()
    next_command = started
    next_encoder = started
    measured_q = start_q.copy()
    last_sent = np.full(5, -1, dtype=int)
    while True:
        now = time.perf_counter()
        progress = min(1.0, (now - started) / duration)
        alpha = minimum_jerk(progress)
        command_q = (1.0 - alpha) * start_q + alpha * goal_q
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
            result = safety.evaluate(measured_q, collision_margin, floor_margin)
            if not result.safe:
                print(f"\n안전 중단({label}): {result.reasons}")
                return measured_q, True
            next_encoder = now + 1.0 / encoder_rate
        tracking = float(np.max(np.abs(np.rad2deg(command_q - measured_q))))
        render(window, safety, scene, context, camera, option,
               measured_q, progress, tracking)
        if glfw.window_should_close(window):
            return measured_q, True
        if progress >= 1.0:
            time.sleep(0.25)
            measured_ticks = read_positions(driver)
            return ticks_to_radians(measured_ticks, centers, signs), False


def hold_and_display(
    driver: MiniFeetechDriver, safety: JDCobotSafetyModel, window,
    scene, context, camera, option, duration: float, centers: np.ndarray,
    signs: np.ndarray, encoder_rate: float,
) -> tuple[np.ndarray, bool]:
    """Keep displaying measured goal pose so small XYZ moves are visible."""
    started = time.perf_counter()
    measured_q = np.zeros(5, dtype=float)
    while time.perf_counter() - started < duration:
        measured_ticks = read_positions(driver)
        measured_q = ticks_to_radians(measured_ticks, centers, signs)
        progress = min(1.0, (time.perf_counter() - started) / duration)
        render(window, safety, scene, context, camera, option,
               measured_q, progress, 0.0)
        if glfw.window_should_close(window):
            return measured_q, True
        time.sleep(1.0 / encoder_rate)
    return measured_q, False


def main() -> int:
    args = parse_args()
    if min(args.ready_time, args.move_time, args.return_time,
           args.encoder_rate) <= 0:
        raise ValueError("이동 시간과 encoder-rate는 양수여야 합니다.")
    if args.goal_hold_time < 0:
        raise ValueError("goal-hold-time은 0 이상이어야 합니다.")
    if args.path_steps < 2 or args.ik_iterations < 1:
        raise ValueError("path-steps는 2 이상, ik-iterations는 1 이상이어야 합니다.")
    if min(args.collision_margin_mm, args.floor_margin_mm,
           args.ik_tolerance_mm, args.min_jacobian_singular_value) < 0:
        raise ValueError("margin, tolerance, 변화 제한 및 특이값 제한은 음수일 수 없습니다.")
    if args.max_ik_joint_change_deg is not None and args.max_ik_joint_change_deg < 0:
        raise ValueError("관절 변화 제한은 음수일 수 없습니다.")
    if args.ik_restarts < 1:
        raise ValueError("ik-restarts는 1 이상이어야 합니다.")
    max_ik_joint_change_deg = (
        args.max_ik_joint_change_deg
        if args.max_ik_joint_change_deg is not None
        else (120.0 if args.target_xyz is not None else 10.0)
    )

    safety = JDCobotSafetyModel()
    signs = parse_signs(args.joint_signs)
    ready_q = np.deg2rad(np.asarray(args.ready_q_deg, dtype=float))
    home_q = np.zeros(5, dtype=float)
    if np.any(ready_q < safety.lower) or np.any(ready_q > safety.upper):
        raise RuntimeError("READY 자세가 MuJoCo 관절 제한 밖입니다.")

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
        if not safety.evaluate(start_q).safe:
            raise RuntimeError("현재 실측 자세가 safe-zone 밖입니다.")

        ready_ticks = validate_ticks(
            ready_q, centers, signs, lower_ticks, upper_ticks, "READY"
        )
        ready_clearance = check_joint_path(safety, start_q, ready_q, args, "READY 진입")
        ready_sigma, ready_condition = jacobian_metrics(safety, ready_q)
        if ready_sigma < args.min_jacobian_singular_value:
            raise RuntimeError(
                f"READY 자세가 특이점에 너무 가깝습니다: sigma_min={ready_sigma:.6f}, "
                f"condition={ready_condition:.1f}"
            )

        ready_tcp = safety.evaluate(ready_q).tcp_position
        target_xyz = (
            np.asarray(args.target_xyz, dtype=float)
            if args.target_xyz is not None
            else ready_tcp + np.asarray(args.delta_xyz_mm, dtype=float) / 1000.0
        )
        if args.target_xyz is not None:
            goal_q, _, iterations = solve_absolute_position_ik(
                safety, ready_q, target_xyz, args
            )
            iteration_label = f"{iterations} safe candidates"
        else:
            goal_q, _, iterations = solve_position_ik(
                safety, ready_q, target_xyz, args.ik_iterations,
                args.ik_tolerance_mm / 1000.0,
            )
            goal_q[4] = ready_q[4]
            iteration_label = f"{iterations} iterations"
        goal_result = safety.evaluate(goal_q)
        ik_error = float(np.linalg.norm(goal_result.tcp_position - target_xyz))
        if ik_error > args.ik_tolerance_mm / 1000.0:
            raise RuntimeError(f"IK 허용 오차 초과: {ik_error * 1000:.2f} mm")
        ik_change = np.rad2deg(goal_q - ready_q)
        if np.any(np.abs(ik_change) > max_ik_joint_change_deg):
            raise RuntimeError(
                f"READY 기준 IK 관절 변화 제한 {max_ik_joint_change_deg:g} deg 초과: "
                f"{np.round(ik_change, 3)}"
            )
        goal_ticks = validate_ticks(
            goal_q, centers, signs, lower_ticks, upper_ticks, "IK 목표"
        )
        goal_clearance = check_joint_path(safety, ready_q, goal_q, args, "IK")
        worst_sigma = check_singularity_path(
            safety, ready_q, goal_q, args.path_steps,
            args.min_jacobian_singular_value,
        )

        return_q = None
        if args.return_to == "ready":
            return_q = ready_q
        elif args.return_to == "home":
            return_q = home_q
        if return_q is not None:
            validate_ticks(return_q, centers, signs, lower_ticks, upper_ticks, "복귀 목표")
            check_joint_path(safety, goal_q, return_q, args, "복귀")

        print("\nHOME/현재 자세 -> READY -> Cartesian IK 사전 검사 통과")
        print("현재 q(deg):", np.round(np.rad2deg(start_q), 3))
        print("READY q(deg):", np.round(np.rad2deg(ready_q), 3), "tick:", ready_ticks)
        print("READY TCP(m):", np.round(ready_tcp, 5))
        print(f"READY sigma_min/condition: {ready_sigma:.6f}/{ready_condition:.1f}")
        print("목표 TCP(m):", np.round(target_xyz, 5), "tick:", goal_ticks)
        print("READY 기준 IK 변화(deg):", np.round(ik_change, 3))
        print(f"IK/FK 오차: {ik_error * 1000:.3f} mm ({iteration_label})")
        print(f"IK 경로 최저 sigma_min: {worst_sigma[0]:.6f} "
              f"(condition={worst_sigma[1]:.1f})")
        print(f"READY/IK 경로 최소 self clearance: "
              f"{ready_clearance[0][0] * 1000:.2f}/"
              f"{goal_clearance[0][0] * 1000:.2f} mm")
        print("작업 후:", args.return_to)
        if input("전체 동작을 시작하려면 MOVE를 입력하세요: ").strip() != "MOVE":
            print("취소했습니다. 모터 명령을 보내지 않았습니다.")
            return 0

        window, scene, context, camera, option = create_window(safety.model)
        for motor_id, tick in zip(MOTOR_IDS, start_ticks):
            driver.set_position(int(motor_id), int(tick))
            if not driver.set_torque_verified(int(motor_id), True):
                raise RuntimeError(f"모터 ID {motor_id} Torque ON 확인 실패")
            time.sleep(0.05)
        torque_enabled = True
        collision_margin = args.collision_margin_mm / 1000.0
        floor_margin = args.floor_margin_mm / 1000.0

        effective_move_time = max(
            args.move_time, float(np.max(np.abs(ik_change))) / 15.0
        )
        if effective_move_time > args.move_time:
            print(f"큰 절대 이동을 위해 IK 이동 시간을 {effective_move_time:.1f}초로 자동 연장합니다.")
        measured_q, stopped = move_segment(
            driver, safety, window, scene, context, camera, option,
            start_q, ready_q, args.ready_time, centers, signs,
            args.encoder_rate, collision_margin, floor_margin, "READY 진입",
        )
        if not stopped:
            measured_q, stopped = move_segment(
                driver, safety, window, scene, context, camera, option,
                measured_q, goal_q, effective_move_time, centers, signs,
                args.encoder_rate, collision_margin, floor_margin, "Cartesian IK",
            )
        if not stopped and args.goal_hold_time > 0:
            measured_q, stopped = hold_and_display(
                driver, safety, window, scene, context, camera, option,
                args.goal_hold_time, centers, signs, args.encoder_rate,
            )
            measured_goal = safety.evaluate(
                measured_q, collision_margin, floor_margin
            ).tcp_position
            xyz_error_mm = (measured_goal - target_xyz) * 1000.0
            print("\nIK 목표 도착 실측 TCP(m):", np.round(measured_goal, 5))
            print("목표 대비 XYZ 오차(mm):", np.round(xyz_error_mm, 3))
            print(f"목표 대비 거리 오차: {np.linalg.norm(xyz_error_mm):.2f} mm")
        if not stopped and return_q is not None:
            measured_q, stopped = move_segment(
                driver, safety, window, scene, context, camera, option,
                measured_q, return_q, args.return_time, centers, signs,
                args.encoder_rate, collision_margin, floor_margin, "복귀",
            )

        if stopped:
            hold_ticks = radians_to_ticks(measured_q, centers, signs)
            for motor_id, tick in zip(MOTOR_IDS, hold_ticks):
                driver.set_position(int(motor_id), int(tick))
            print("이동을 중단하고 마지막 실측 위치를 유지합니다.")
            return 2
        final_result = safety.evaluate(measured_q, collision_margin, floor_margin)
        print("\n전체 동작 완료")
        print("최종 실측 q(deg):", np.round(np.rad2deg(measured_q), 3))
        print("최종 실측 TCP(m):", np.round(final_result.tcp_position, 5))
        if not args.release_after_move:
            print("현재 자세 유지를 위해 토크를 켠 상태로 종료합니다.")
        return 0
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
