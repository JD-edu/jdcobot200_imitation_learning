#!/usr/bin/env python3
"""실물과 MuJoCo의 관절 방향을 비교하고 robot.yaml을 갱신합니다."""

# 실행 예:
# /usr/bin/python3 projects/curriculum/04_sim2real_fk/01_calibrate_joint_direction.py --joint 1
# /usr/bin/python3 projects/curriculum/04_sim2real_fk/01_calibrate_joint_direction.py --joint wrist_pitch --test-deg 10

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml

from sim2real_control import MuJoCoViewer, SynchronizedController, resolve_joint


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config/jdcobot200/robot.yaml"


def update_direction(joint_name: str, direction: int) -> None:
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    target = next((joint for joint in data["joints"] if joint["name"] == joint_name), None)
    if target is None:
        raise ValueError(f"robot.yaml에 관절이 없습니다: {joint_name}")
    target["direction"] = int(direction)
    backup = CONFIG_PATH.with_suffix(".yaml.bak")
    shutil.copy2(CONFIG_PATH, backup)
    temporary = CONFIG_PATH.with_suffix(".yaml.tmp")
    temporary.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    temporary.replace(CONFIG_PATH)
    print(f"robot.yaml 갱신 완료: {joint_name}.direction={direction}")
    print(f"이전 설정 백업: {backup}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="작은 시험 이동으로 실제/MuJoCo 관절 방향을 비교합니다."
    )
    parser.add_argument("--joint", required=True, help="1~5 또는 관절 이름")
    parser.add_argument("--test-deg", type=float, default=10.0, help="시험 이동 크기(기본·최대 10도)")
    parser.add_argument("--move-time", type=float, default=2.0)
    parser.add_argument("--port")
    args = parser.parse_args()
    if not 0 < args.test_deg <= 10:
        raise ValueError("--test-deg는 0보다 크고 10도 이하여야 합니다.")

    index = resolve_joint(args.joint)
    viewer = None
    with SynchronizedController(port=args.port) as controller:
        joint = controller.config.arm_joints[index]
        start_raw, start_q = controller.read(samples=5)
        corrected_ticks = {
            arm_joint.name: start_raw[arm_joint.name] + arm_joint.home_offset_tick
            for arm_joint in controller.config.arm_joints
        }
        tick_delta = max(1, round(args.test_deg * controller.config.protocol_resolution / 360.0))
        raw_sign = 1 if start_raw[joint.name] + tick_delta <= joint.max_tick else -1
        test_raw = dict(start_raw)
        test_raw[joint.name] += raw_sign * tick_delta
        test_q = controller.ticks_to_q(test_raw)
        controller.check_path(start_q, test_q)
        controller.check_path(test_q, start_q)

        # 반드시 실제 encoder를 읽은 뒤 그 값으로 MuJoCo의 첫 자세를 잡습니다.
        viewer = MuJoCoViewer(controller.safety, f"ENCODER START: {joint.name}")
        viewer.render(start_q, f"ENCODER START | selected {index + 1}: {joint.name}")

        print(f"선택 관절: {index + 1} ({joint.name})")
        print("실제 encoder raw:", [start_raw[name] for name in controller.config.joint_names[:5]])
        print("raw + offset:", [corrected_ticks[name] for name in controller.config.joint_names[:5]])
        print("MuJoCo 초기 q(deg):", np.round(np.rad2deg(start_q), 2))
        print(f"현재 direction: {joint.direction:+d}")
        print(f"실제 시험: raw tick {start_raw[joint.name]} -> {test_raw[joint.name]}")
        print(f"MuJoCo 시험각: {np.rad2deg(start_q[index]):.2f} -> {np.rad2deg(test_q[index]):.2f} deg")
        print("왕복 경로 안전검사를 통과했습니다. 주변과 비상 정지 수단을 확인하세요.")
        if input("실물과 MuJoCo 방향 비교를 시작하려면 MOVE 입력: ").strip() != "MOVE":
            print("취소했습니다. 설정과 모터를 변경하지 않았습니다.")
            viewer.close()
            return 0

        try:
            # 첫 명령 전에 encoder 초기 자세를 한 번 더 확실히 표시합니다.
            viewer.render(start_q, f"STARTING FROM ENCODER POSE: {joint.name}")
            _, reached_q = controller.move_to(
                start_raw, start_q, test_q, duration=args.move_time,
                on_step=lambda q: viewer.render(q, f"COMPARE DIRECTION: {joint.name}"),
            )
            input("두 로봇의 움직임을 관찰했으면 Enter를 눌러 원위치합니다: ")
            measured_raw, _ = controller.read(samples=3)
            controller.move_to(
                measured_raw, reached_q, start_q, duration=args.move_time,
                on_step=lambda q: viewer.render(q, f"RETURN: {joint.name}"),
            )
        finally:
            viewer.close()

        answer = input("실물 방향이 화면의 MuJoCo와 같으면 SAME, 반대면 OPPOSITE 입력: ").strip().upper()
        if answer == "SAME":
            new_direction = joint.direction
        elif answer == "OPPOSITE":
            new_direction = -joint.direction
        else:
            print("판정을 저장하지 않았습니다. robot.yaml은 변경되지 않았습니다.")
            return 0
        if input(f"{joint.name}.direction={new_direction:+d} 저장은 UPDATE 입력: ").strip() != "UPDATE":
            print("저장을 취소했습니다. robot.yaml은 변경되지 않았습니다.")
            return 0
        update_direction(joint.name, new_direction)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
