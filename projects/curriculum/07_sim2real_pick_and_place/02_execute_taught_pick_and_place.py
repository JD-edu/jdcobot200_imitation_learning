#!/usr/bin/env python3
"""손으로 티칭한 Cartesian 지점을 실제 로봇의 IK pick-and-place로 실행합니다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LESSON_DIR = Path(__file__).resolve().parent
COURSE04 = PROJECT_ROOT / "projects/curriculum/04_sim2real_fk"
COURSE05 = PROJECT_ROOT / "projects/curriculum/05_mujoco_ik"
COURSE06 = PROJECT_ROOT / "projects/curriculum/06_sim2real_ik"
for path in (PROJECT_ROOT, COURSE04, COURSE05, COURSE06):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ik_solver import SafeIK  # noqa: E402
from sim2real_control import MuJoCoViewer, SynchronizedController  # noqa: E402
from target_ik_common import (  # noqa: E402
    SafeCandidate,
    configure_ik_limits,
    execute_candidate,
    print_candidate,
    read_safe_start,
    validate_candidate,
)


ARM_SEQUENCE = ("pick_ready", "lift", "place_approach", "place")
REQUIRED_POINTS = (
    "pick_ready", "grasp", "lift", "place_approach", "place", "release"
)


def load_teaching(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"티칭 파일이 없습니다: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"티칭 JSON 형식 오류: {exc}") from exc
    if payload.get("format") != "jdcobot200_cartesian_pick_place_teaching":
        raise ValueError("지원하지 않는 티칭 파일 format입니다.")
    points = payload.get("points")
    if not isinstance(points, dict):
        raise ValueError("티칭 파일에 points 객체가 없습니다.")
    missing = [name for name in REQUIRED_POINTS if name not in points]
    if missing:
        raise ValueError("티칭 지점 누락: " + ", ".join(missing))
    for name in REQUIRED_POINTS:
        xyz = np.asarray(points[name].get("tcp_xyz_m"), dtype=float)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError(f"{name}: 유효한 tcp_xyz_m 좌표가 없습니다.")
        if not isinstance(points[name].get("gripper_raw_tick"), int):
            raise ValueError(f"{name}: gripper_raw_tick이 없습니다.")
    return payload


def plan_arm_sequence(
    controller: SynchronizedController,
    solver: SafeIK,
    start_q: np.ndarray,
    points: dict,
    *,
    tolerance_mm: float,
    max_joint_change_deg: float,
    path_steps: int,
) -> dict[str, SafeCandidate]:
    """실제 명령 전에 모든 arm waypoint의 IK와 경로를 순서대로 검사한다."""
    planned = {}
    seed_q = start_q.copy()
    for name in ARM_SEQUENCE:
        target = np.asarray(points[name]["tcp_xyz_m"], dtype=float)
        try:
            ik = solver.solve(seed_q, target, tolerance_m=tolerance_mm / 1000.0)
            candidate = validate_candidate(
                controller,
                solver,
                seed_q,
                ik,
                max_joint_change_deg=max_joint_change_deg,
                path_steps=path_steps,
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            raise RuntimeError(f"{name} 사전계획 실패: {exc}") from exc
        planned[name] = candidate
        seed_q = candidate.ik.q_rad
    return planned


def move_gripper(
    controller: SynchronizedController,
    target_tick: int,
    *,
    duration: float,
    feedback_tolerance_ticks: int,
) -> int:
    """현재 위치를 먼저 hold한 뒤 티칭한 gripper tick으로 저속 이동한다."""
    joint = controller.config.gripper_joint
    controller.robot.validate_raw_target(joint, target_tick)
    measured = controller.robot.driver.get_position_filtered(joint.motor_id, samples=7)
    if measured is None:
        raise RuntimeError("gripper encoder 읽기 실패")
    start = {joint.name: int(measured)}
    controller.robot.validate_raw_target(joint, start[joint.name])
    controller.robot.hold_and_enable(start, (joint,))
    controller.robot.interpolate_raw(
        start,
        {joint.name: target_tick},
        duration=duration,
        rate_hz=20.0,
        max_step_ticks=12,
    )
    reached = controller.robot.wait_for_raw_targets(
        {joint.name: target_tick},
        tolerance=feedback_tolerance_ticks,
        timeout=2.0,
    )
    return reached[joint.name]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="티칭 JSON의 Cartesian 좌표로 실제 pick-and-place를 실행합니다."
    )
    parser.add_argument(
        "--input", type=Path,
        default=LESSON_DIR / "taught_pick_place.json",
        help="01 코드로 만든 티칭 JSON",
    )
    parser.add_argument("--arm-move-time", type=float, default=3.0)
    parser.add_argument("--gripper-move-time", type=float, default=1.0)
    parser.add_argument("--settle-time", type=float, default=0.5)
    parser.add_argument("--tolerance-mm", type=float, default=1.0)
    parser.add_argument("--max-joint-change-deg", type=float, default=120.0)
    parser.add_argument("--feedback-tolerance-ticks", type=int, default=50)
    parser.add_argument("--path-steps", type=int, default=201)
    parser.add_argument("--no-retreat", action="store_true",
                        help="release 후 place_approach로 상승하지 않음")
    parser.add_argument("--port")
    args = parser.parse_args()
    if (args.arm_move_time <= 0 or args.gripper_move_time <= 0
            or args.settle_time < 0 or args.tolerance_mm <= 0
            or args.max_joint_change_deg <= 0):
        raise ValueError("이동시간과 tolerance는 양수이고 settle-time은 0 이상이어야 합니다.")
    if not 1 <= args.feedback_tolerance_ticks <= 100 or args.path_steps < 2:
        raise ValueError("feedback tolerance는 1~100 tick, path steps는 2 이상이어야 합니다.")

    teaching = load_teaching(args.input.expanduser().resolve())
    points = teaching["points"]
    with SynchronizedController(port=args.port) as controller:
        configure_ik_limits(controller)
        solver = SafeIK(controller.safety)
        current_raw, current_q = read_safe_start(controller)

        # 원점은 position IK의 특이 자세이므로 항상 설정 파일의 READY를 거친다.
        ready_q = np.deg2rad(
            np.asarray(controller.config.ready_pose_degrees, dtype=float)
        )
        ready_result = controller.safety.evaluate(ready_q)
        if not ready_result.safe:
            raise RuntimeError(f"READY 자세가 안전하지 않습니다: {ready_result.reasons}")
        controller.check_path(current_q, ready_q, steps=args.path_steps)
        controller.q_to_ticks(ready_q)

        # 어떤 torque도 켜기 전에 전체 pick-and-place arm 경로를 먼저 계획한다.
        planned = plan_arm_sequence(
            controller,
            solver,
            ready_q,
            points,
            tolerance_mm=args.tolerance_mm,
            max_joint_change_deg=args.max_joint_change_deg,
            path_steps=args.path_steps,
        )
        gripper = controller.config.gripper_joint
        open_pick_tick = int(points["pick_ready"]["gripper_raw_tick"])
        grasp_tick = int(points["grasp"]["gripper_raw_tick"])
        release_tick = int(points["release"]["gripper_raw_tick"])
        for tick in (open_pick_tick, grasp_tick, release_tick):
            controller.robot.validate_raw_target(gripper, tick)

        print("티칭 파일:", args.input.expanduser().resolve())
        print("현재 q(deg):", np.round(np.rad2deg(current_q), 3))
        print("먼저 이동할 READY q(deg):", np.round(np.rad2deg(ready_q), 3))
        for name in ARM_SEQUENCE:
            print_candidate(name, planned[name])
        print(
            "gripper ticks: "
            f"open={open_pick_tick}, grasp={grasp_tick}, release={release_tick}"
        )
        print("전체 IK/FK/경로/hardware 범위 사전검사를 통과했습니다.")
        if input("실제 pick-and-place를 실행하려면 RUN 입력: ").strip() != "RUN":
            print("취소했습니다. 모터 명령을 보내지 않았습니다.")
            return 0

        viewer = MuJoCoViewer(controller.safety, "Taught Sim2Real Pick-and-Place")
        try:
            print("[1/8] IK 특이점을 피하기 위해 READY 자세로 이동합니다.")
            try:
                current_raw, current_q = controller.move_to(
                    current_raw,
                    current_q,
                    ready_q,
                    duration=args.arm_move_time,
                    feedback_tolerance_ticks=args.feedback_tolerance_ticks,
                    on_step=lambda q: viewer.render(q, "Moving to READY pose"),
                )
            except TimeoutError:
                current_raw, current_q = controller.read(samples=7)
                measured = controller.safety.evaluate(current_q)
                if not measured.safe:
                    raise RuntimeError(
                        f"READY 이동 타임아웃 후 자세가 안전하지 않습니다: "
                        f"{measured.reasons}"
                    )
                print(
                    "경고: READY 목표 도달 타임아웃. 실제 자세로 재동기화합니다.",
                    "오차(deg)=", np.round(np.rad2deg(ready_q - current_q), 3),
                )
            time.sleep(args.settle_time)

            print("[2/8] gripper를 엽니다.")
            move_gripper(
                controller, open_pick_tick,
                duration=args.gripper_move_time,
                feedback_tolerance_ticks=args.feedback_tolerance_ticks,
            )
            time.sleep(args.settle_time)

            def move_arm(name: str, step: str) -> None:
                nonlocal current_raw, current_q
                # 실제 추종 오차가 있으므로 현재 측정 자세에서 같은 목표를 재검사한다.
                candidate = validate_candidate(
                    controller,
                    solver,
                    current_q,
                    planned[name].ik,
                    max_joint_change_deg=args.max_joint_change_deg,
                    path_steps=args.path_steps,
                )
                print(f"[{step}] {name} 이동")
                current_raw, current_q = execute_candidate(
                    controller,
                    viewer,
                    current_raw,
                    current_q,
                    candidate,
                    move_time=args.arm_move_time,
                    feedback_tolerance_ticks=args.feedback_tolerance_ticks,
                )
                time.sleep(args.settle_time)

            move_arm("pick_ready", "3/8")

            print("[4/8] gripper를 닫아 물체를 잡습니다.")
            move_gripper(
                controller, grasp_tick,
                duration=args.gripper_move_time,
                feedback_tolerance_ticks=args.feedback_tolerance_ticks,
            )
            time.sleep(args.settle_time)

            move_arm("lift", "5/8")
            move_arm("place_approach", "6/8")
            move_arm("place", "7/8")

            print("[8/8] gripper를 열어 물체를 놓습니다.")
            move_gripper(
                controller, release_tick,
                duration=args.gripper_move_time,
                feedback_tolerance_ticks=args.feedback_tolerance_ticks,
            )
            time.sleep(args.settle_time)

            if not args.no_retreat:
                move_arm("place_approach", "후퇴")
            final = controller.safety.evaluate(current_q)
            print("pick-and-place 완료")
            print("최종 TCP(mm):", np.round(final.tcp_position * 1000.0, 3))
            print("arm과 gripper torque는 ON 상태입니다.")
        finally:
            viewer.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, TimeoutError, ValueError,
            np.linalg.LinAlgError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
