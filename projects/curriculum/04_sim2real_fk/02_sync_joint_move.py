#!/usr/bin/env python3
"""입력한 한 관절을 실제 로봇과 MuJoCo에서 안전하게 동기 이동합니다."""

# 실행 예:
# /usr/bin/python3 projects/curriculum/04_sim2real_fk/02_sync_joint_move.py --joint 2 --delta-deg 3
# /usr/bin/python3 projects/curriculum/04_sim2real_fk/02_sync_joint_move.py --joint wrist_pitch --target-deg 10
# /usr/bin/python3 projects/curriculum/04_sim2real_fk/02_sync_joint_move.py --all --delta-deg 2 -3 4 -2 1
# /usr/bin/python3 projects/curriculum/04_sim2real_fk/02_sync_joint_move.py --all --target-deg 0 20 -40 20 0

import argparse
import sys

import numpy as np

from sim2real_control import MuJoCoViewer, SynchronizedController, resolve_joint


def main() -> int:
    parser = argparse.ArgumentParser(description="한 관절 또는 전체 관절의 실제/MuJoCo 동기 이동")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--joint", help="1~5 또는 관절 이름")
    selection.add_argument("--all", action="store_true", help="arm 관절 5개를 동시에 이동")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--delta-deg", nargs="+", type=float,
        help="현재 자세 기준 상대각: --joint는 1개, --all은 5개",
    )
    target.add_argument(
        "--target-deg", nargs="+", type=float,
        help="MuJoCo 논리 절대각: --joint는 1개, --all은 5개",
    )
    parser.add_argument(
        "--max-change-deg", type=float,
        help="관절별 1회 변화 제한(기본값: --joint 10도, --all 45도)",
    )
    parser.add_argument("--move-time", type=float, default=3.0)
    parser.add_argument("--port")
    args = parser.parse_args()
    if args.max_change_deg is not None and args.max_change_deg <= 0:
        raise ValueError("max-change-deg는 양수여야 합니다.")
    if args.move_time <= 0:
        raise ValueError("move-time은 양수여야 합니다.")

    values = args.delta_deg if args.delta_deg is not None else args.target_deg
    expected = 5 if args.all else 1
    max_change_deg = args.max_change_deg if args.max_change_deg is not None else (45.0 if args.all else 10.0)
    if len(values) != expected:
        raise ValueError(
            f"{'--all' if args.all else '--joint'}에는 각도 {expected}개가 필요합니다."
        )
    indices = list(range(5)) if args.all else [resolve_joint(args.joint)]
    with SynchronizedController(port=args.port) as controller:
        start_raw, start_q = controller.read(samples=5)
        goal_q = start_q.copy()
        values_rad = np.deg2rad(np.asarray(values, dtype=float))
        for value_index, joint_index in enumerate(indices):
            goal_q[joint_index] = (
                start_q[joint_index] + values_rad[value_index]
                if args.delta_deg is not None else values_rad[value_index]
            )
        changes = np.rad2deg(goal_q - start_q)
        if np.max(np.abs(changes)) > max_change_deg:
            raise ValueError(
                f"요청 변화량 {np.round(changes, 2)}도가 관절별 제한 "
                f"{max_change_deg:.2f}도를 초과합니다."
            )
        controller.check_path(start_q, goal_q)
        goal_raw = controller.q_to_ticks(goal_q)
        move_label = "arm 1-5" if args.all else controller.config.arm_joints[indices[0]].name

        print("관절 순서:", ", ".join(controller.config.joint_names[:5]))
        print("시작각(deg):", np.round(np.rad2deg(start_q), 2))
        print("목표각(deg):", np.round(np.rad2deg(goal_q), 2))
        print("변화량(deg):", np.round(changes, 2))
        print(f"관절별 변화 제한: {max_change_deg:.1f} deg")
        print("시작 raw:", [start_raw[name] for name in controller.config.joint_names[:5]])
        print("목표 raw:", [goal_raw[name] for name in controller.config.joint_names[:5]])
        print("전체 경로 안전검사를 통과했습니다. 실제 로봇은 아직 움직이지 않았습니다.")
        if input("두 로봇을 동기 이동하려면 MOVE 입력: ").strip() != "MOVE":
            print("취소했습니다. 모터 명령을 보내지 않았습니다.")
            return 0

        viewer = MuJoCoViewer(controller.safety, f"Synchronized move: {move_label}")
        try:
            _, measured_q = controller.move_to(
                start_raw, start_q, goal_q, duration=args.move_time,
                on_step=lambda q: viewer.render(
                    q, f"SYNC {move_label}: {np.round(np.rad2deg(q), 1)} deg"
                ),
            )
            viewer.render(measured_q, f"DONE {move_label} | ESC or close window")
            print("동기 이동을 완료했습니다. 창을 닫거나 ESC를 누르세요.")
            while not viewer.should_close():
                viewer.render(measured_q)
        finally:
            viewer.close()
        print("자세 유지를 위해 arm torque는 ON 상태입니다.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
