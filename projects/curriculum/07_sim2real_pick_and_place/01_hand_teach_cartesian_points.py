#!/usr/bin/env python3
"""팔을 손으로 움직여 pick/place TCP 좌표를 안전하게 기록합니다."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LESSON_DIR = Path(__file__).resolve().parent
COURSE04 = PROJECT_ROOT / "projects/curriculum/04_sim2real_fk"
COURSE05 = PROJECT_ROOT / "projects/curriculum/05_mujoco_ik"
COURSE06 = PROJECT_ROOT / "projects/curriculum/06_sim2real_ik"
for path in (PROJECT_ROOT, COURSE04, COURSE05, COURSE06):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sim2real_control import JOINT_NAMES, SynchronizedController  # noqa: E402
from target_ik_common import configure_ik_limits  # noqa: E402


POINTS = (
    (
        "pick_ready",
        "열린 그리퍼를 물체를 잡을 위치에 놓고 정지하세요.",
    ),
    (
        "grasp",
        "TCP 위치를 유지한 채 그리퍼를 손으로 닫아 물체를 잡으세요.",
    ),
    (
        "lift",
        "물체를 잡은 상태로 충돌하지 않을 안전 높이까지 들어 올리세요.",
    ),
    (
        "place_approach",
        "물체를 놓을 목표 지점의 상공으로 이동해 정지하세요.",
    ),
    (
        "place",
        "물체를 목표 지점의 바닥까지 천천히 내리세요.",
    ),
    (
        "release",
        "TCP 위치를 유지한 채 그리퍼를 손으로 열어 물체를 완전히 분리하세요.",
    ),
)

MIN_VERTICAL_CLEARANCE_MM = 10.0
MAX_GRIPPER_ACTION_TCP_DRIFT_MM = 20.0


def capture_point(controller: SynchronizedController, name: str) -> dict:
    """현재 encoder 자세를 읽고 안전한 FK/TCP 정보만 반환합니다."""
    raw, q_rad = controller.read(samples=7)
    result = controller.safety.evaluate(q_rad)
    if not result.safe:
        raise RuntimeError(
            f"현재 자세는 저장할 수 없습니다: {result.reasons}, "
            f"q(deg)={np.round(np.rad2deg(q_rad), 3)}"
        )
    # 기록한 자세가 실제 하드웨어 명령 범위 안인지도 확인한다.
    controller.q_to_ticks(q_rad)
    gripper = controller.config.gripper_joint
    gripper_raw = controller.robot.driver.get_position_filtered(
        gripper.motor_id, samples=7
    )
    if gripper_raw is None:
        raise RuntimeError("gripper encoder 읽기 실패")
    controller.robot.validate_raw_target(gripper, int(gripper_raw))
    return {
        "name": name,
        "tcp_xyz_m": np.round(result.tcp_position, 9).tolist(),
        "tcp_xyz_mm": np.round(result.tcp_position * 1000.0, 6).tolist(),
        "tcp_rotation_matrix": np.round(result.tcp_matrix, 9).tolist(),
        "joint_names": list(JOINT_NAMES),
        "joint_degrees": np.round(np.rad2deg(q_rad), 6).tolist(),
        "raw_ticks": raw,
        "gripper_raw_tick": int(gripper_raw),
        "self_clearance_mm": round(result.self_clearance * 1000.0, 6),
        "floor_clearance_mm": round(result.floor_clearance * 1000.0, 6),
    }


def save_json(path: Path, payload: dict) -> None:
    """완성된 teaching 결과를 같은 폴더에서 원자적으로 저장합니다."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_teaching_sequence(captures: list[dict]) -> None:
    """파지/해제 위치 유지와 수직 안전 높이를 검사한다."""
    points = {point["name"]: point for point in captures}

    def xyz(name: str) -> np.ndarray:
        return np.asarray(points[name]["tcp_xyz_mm"], dtype=float)

    grasp_drift = float(np.linalg.norm(xyz("grasp") - xyz("pick_ready")))
    if grasp_drift > MAX_GRIPPER_ACTION_TCP_DRIFT_MM:
        raise RuntimeError(
            f"그리퍼를 닫는 동안 TCP가 {grasp_drift:.2f} mm 이동했습니다. "
            f"허용값은 {MAX_GRIPPER_ACTION_TCP_DRIFT_MM:g} mm입니다."
        )

    release_drift = float(np.linalg.norm(xyz("release") - xyz("place")))
    if release_drift > MAX_GRIPPER_ACTION_TCP_DRIFT_MM:
        raise RuntimeError(
            f"그리퍼를 여는 동안 TCP가 {release_drift:.2f} mm 이동했습니다. "
            f"허용값은 {MAX_GRIPPER_ACTION_TCP_DRIFT_MM:g} mm입니다."
        )

    lift_clearance = float(xyz("lift")[2] - xyz("grasp")[2])
    if lift_clearance < MIN_VERTICAL_CLEARANCE_MM:
        raise RuntimeError(
            f"lift 높이가 grasp보다 {lift_clearance:.2f} mm만 높습니다. "
            f"최소 {MIN_VERTICAL_CLEARANCE_MM:g} mm 이상 올려야 합니다."
        )

    place_clearance = float(xyz("place_approach")[2] - xyz("place")[2])
    if place_clearance < MIN_VERTICAL_CLEARANCE_MM:
        raise RuntimeError(
            f"place_approach 높이가 place보다 {place_clearance:.2f} mm만 높습니다. "
            f"최소 {MIN_VERTICAL_CLEARANCE_MM:g} mm 이상이어야 합니다."
        )


def validate_new_capture(previous: list[dict], point: dict) -> None:
    """현재 단계와 직전 관련 단계의 기하 조건을 즉시 검사한다."""
    saved = {item["name"]: item for item in previous}
    current = np.asarray(point["tcp_xyz_mm"], dtype=float)
    name = point["name"]

    if name == "grasp":
        reference = np.asarray(saved["pick_ready"]["tcp_xyz_mm"], dtype=float)
        drift = float(np.linalg.norm(current - reference))
        if drift > MAX_GRIPPER_ACTION_TCP_DRIFT_MM:
            raise RuntimeError(
                f"그리퍼를 닫는 동안 TCP가 {drift:.2f} mm 이동했습니다. "
                "팔 위치를 다시 맞추세요."
            )
    elif name == "lift":
        reference = np.asarray(saved["grasp"]["tcp_xyz_mm"], dtype=float)
        clearance = float(current[2] - reference[2])
        if clearance < MIN_VERTICAL_CLEARANCE_MM:
            raise RuntimeError(
                f"lift 높이 차이가 {clearance:.2f} mm입니다. "
                "물체를 더 높이 올리세요."
            )
    elif name == "place":
        reference = np.asarray(saved["place_approach"]["tcp_xyz_mm"], dtype=float)
        clearance = float(reference[2] - current[2])
        if clearance < MIN_VERTICAL_CLEARANCE_MM:
            raise RuntimeError(
                f"place 하강 거리가 {clearance:.2f} mm입니다. "
                "물체를 목표 바닥까지 더 내리세요."
            )
    elif name == "release":
        reference = np.asarray(saved["place"]["tcp_xyz_mm"], dtype=float)
        drift = float(np.linalg.norm(current - reference))
        if drift > MAX_GRIPPER_ACTION_TCP_DRIFT_MM:
            raise RuntimeError(
                f"그리퍼를 여는 동안 TCP가 {drift:.2f} mm 이동했습니다. "
                "팔 위치를 다시 맞추세요."
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="팔 torque를 끄고 손으로 움직여 pick/place TCP 좌표를 기록합니다."
    )
    parser.add_argument(
        "--output", type=Path,
        default=LESSON_DIR / "taught_pick_place.json",
        help="저장할 JSON 경로",
    )
    parser.add_argument("--port")
    args = parser.parse_args()

    with SynchronizedController(port=args.port) as controller:
        configure_ik_limits(controller)
        initial_raw, initial_q = controller.read(samples=7)
        initial = controller.safety.evaluate(initial_q)
        if not initial.safe:
            raise RuntimeError(
                f"시작 encoder 자세가 안전하지 않습니다: {initial.reasons}, "
                f"q(deg)={np.round(np.rad2deg(initial_q), 3)}"
            )

        print("손으로 움직일 공간에서 사람과 장애물을 치우고 로봇을 지지하세요.")
        print("arm 5개 관절과 gripper의 torque를 모두 해제합니다.")
        print("시작 q(deg):", np.round(np.rad2deg(initial_q), 3))
        print("시작 TCP(mm):", np.round(initial.tcp_position * 1000.0, 3))
        if input("핸드 티칭을 시작하려면 TEACH 입력: ").strip() != "TEACH":
            print("취소했습니다. torque를 변경하지 않았습니다.")
            return 0

        controller.robot.disable_torque(controller.config.joints)
        torque_states = controller.robot.read_torque_states()
        all_states = {
            joint.name: torque_states[joint.name]
            for joint in controller.config.joints
        }
        if not all(state is False for state in all_states.values()):
            raise RuntimeError(
                f"전체 torque 최종 확인 실패: {all_states}. "
                "일부 모터는 이미 OFF일 수 있으므로 로봇을 계속 지지하세요."
            )
        print(f"arm과 gripper torque를 모두 해제했습니다: {all_states}")
        print("로봇을 항상 손으로 지지하세요.")

        captures = []
        for name, description in POINTS:
            while True:
                print(f"\n[{name}] {description}")
                command = input(
                    "로봇을 손으로 이동한 뒤 CAPTURE 입력 "
                    "(현재 좌표 확인은 SHOW, 종료는 ABORT): "
                ).strip().upper()
                if command == "ABORT":
                    print("티칭을 중단했습니다. arm과 gripper torque는 OFF 상태입니다.")
                    return 1
                if command not in {"CAPTURE", "SHOW"}:
                    print("CAPTURE, SHOW 또는 ABORT를 입력하세요.")
                    continue
                try:
                    point = capture_point(controller, name)
                except (RuntimeError, ValueError) as exc:
                    print("거부:", exc)
                    continue
                print(
                    f"{name} TCP(mm)={point['tcp_xyz_mm']}, "
                    f"q(deg)={point['joint_degrees']}"
                )
                print(
                    "안전 여유 self/floor(mm)="
                    f"{point['self_clearance_mm']}/"
                    f"{point['floor_clearance_mm']}"
                )
                if command == "SHOW":
                    continue
                try:
                    validate_new_capture(captures, point)
                except RuntimeError as exc:
                    print("거부:", exc)
                    continue
                captures.append(point)
                break

        validate_teaching_sequence(captures)
        payload = {
            "format": "jdcobot200_cartesian_pick_place_teaching",
            "version": 2,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "coordinate_frame": "MuJoCo world / robot base",
            "position_unit": "meter",
            "sequence": [name for name, _description in POINTS],
            "points": {point["name"]: point for point in captures},
        }
        save_json(args.output, payload)
        print(f"\n티칭 좌표를 저장했습니다: {args.output.expanduser().resolve()}")

        final_raw, final_q = controller.read(samples=7)
        print("현재 로봇을 손으로 지지한 상태에서 선택하세요.")
        if input("현재 자세를 유지하도록 torque를 켜려면 HOLD 입력: ").strip() == "HOLD":
            final_result = controller.safety.evaluate(final_q)
            if not final_result.safe:
                raise RuntimeError(
                    f"현재 자세가 안전하지 않아 torque를 켜지 않습니다: "
                    f"{final_result.reasons}"
                )
            gripper = controller.config.gripper_joint
            gripper_raw = controller.robot.driver.get_position_filtered(
                gripper.motor_id, samples=7
            )
            if gripper_raw is None:
                raise RuntimeError(
                    "gripper encoder를 읽지 못해 torque를 켜지 않습니다. "
                    "로봇을 계속 지지하세요."
                )
            controller.robot.validate_raw_target(gripper, int(gripper_raw))
            all_raw = dict(final_raw)
            all_raw[gripper.name] = int(gripper_raw)
            controller.robot.hold_and_enable(all_raw, controller.config.joints)
            print("현재 encoder 위치를 목표로 설정한 뒤 arm과 gripper torque를 켰습니다.")
        else:
            print("arm과 gripper torque는 OFF 상태입니다. 로봇을 계속 지지하세요.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
